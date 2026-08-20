"""High-level orchestrator for the legal assistant (Responses API + tools + modes)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from config.model_routing import SUMMARY_MODEL

from agents.models import (
    ChatHistory,
    ErrorEvent,
    HeartbeatEvent,
    ModeName,
    ModeResult,
    QualityModeName,
    SummaryPayload,
    StatusEvent,
    StreamEvent,
    ToolCall,
    TrackedUsage,
    UsageCall,
    settings,
)
from agents.locale import load_prompts as _load_prompts
from agents.shared import (
    StructuredOutputError,
    _count_message_tokens,
    _count_tokens,
    _extract_message_text,
    _get_encoder,
    _preview,
    _strip_trailing_json,
    _text_format,
    _ui_user_turn_count,
    _validate_json_model,
    _with_heartbeats,
)
from agents.search import LegalSearchClient
from agents.mode_base import Mode
from agents.decide import DecideMode
from agents.retrieve import RetrieveMode
from agents.answer import AnswerMode
from agents.clarify import ClarifyMode
from agents.generate import GenerateMode
from agents.user_doc import UserDocRetrieveMode
from agents.process_docs import ProcessDocumentsMode

logger = logging.getLogger(__name__)


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def _retry_delay_seconds(attempt: int, base_delay: float) -> float:
    return base_delay * (attempt + 1)




class ChatManager:
    """High-level orchestrator for the legal assistant (Responses API + tools + modes)."""

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None) -> None:
        self.openai_client = openai_client or AsyncOpenAI(
            timeout=httpx.Timeout(settings.stream_total_timeout_seconds, connect=10.0),
        )
        self.search_url = settings.search_api_url.rstrip("/") + "/"
        self.legal_search = LegalSearchClient(
            search_url=self.search_url,
            alpha=settings.search_alpha,
            timeout_seconds=settings.search_timeout_seconds,
        )
        self.modes: Dict[ModeName, Mode] = {
            mode.name: mode for mode in [
                DecideMode(),
                RetrieveMode(),
                UserDocRetrieveMode(),
                ProcessDocumentsMode(),
                GenerateMode(),
                AnswerMode(),
                ClarifyMode(),
            ]
        }
        logger.info(
            "chat settings openai_model=%s search_url=%s max_docs_per_type=%s max_chunks_per_doc=%s search_chunks=%s search_alpha=%s max_retrieval_passes=%s max_retrieval_refines=%s",
            settings.openai_model,
            self.search_url,
            settings.max_docs_per_type,
            settings.max_chunks_per_doc,
            settings.search_chunks,
            settings.search_alpha,
            settings.max_retrieval_passes,
            settings.max_retrieval_refines,
        )

    async def aclose(self) -> None:
        await self.legal_search.aclose()
        await self.openai_client.close()

    async def handle_message_streaming(
        self, user_message: str, history: Optional[ChatHistory] = None,
        quality_mode: QualityModeName = "thorough",
        append_user_message: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """
        Streaming version of handle_message that yields events as processing happens.
        
        Args:
            user_message: The user's message
            history: Optional existing chat history
            quality_mode: "fast" for speed/cost, "thorough" for quality
            append_user_message: If False, skip appending user message to history
                (caller already persisted it and _build_history loaded it)
        
        Yields:
            StatusEvent: Progress updates
            ChunkEvent: Streaming text chunks from the final answer
            ErrorEvent: If an error occurs
        """
        chat_history = history or ChatHistory()
        
        # Store quality mode in metadata for use by modes
        chat_history.metadata.quality_mode = quality_mode
        
        # Storage safety valve: prune very old messages if over hard cap
        self._prune_old_messages(chat_history)
        
        # Reset per-message rate limiters (allows fresh retrieval each turn)
        chat_history.metadata.mode_steps = 0
        chat_history.metadata.retrieval_calls = 0
        
        # Evict legal sources not cited in the last N user-visible messages
        self._evict_stale_sources(chat_history)
        
        if append_user_message:
            chat_history.ui_chat_history_raw.new_message(role="user", content=user_message)
            chat_history.llm_chat_history_raw.new_message(role="user", content=user_message)

        mode: ModeName = "decide"
        logger.info(
            "handle_message_streaming start mode=%s ui_messages=%s llm_messages=%s user=%s",
            mode,
            len(chat_history.ui_chat_history_raw.conversation_history),
            len(chat_history.llm_chat_history_raw.conversation_history),
            _preview(user_message, 120),
        )

        try:
            while True:
                chat_history.metadata.mode_steps += 1
                if chat_history.metadata.mode_steps > settings.max_mode_steps:
                    yield ErrorEvent(detail=f"Exceeded max mode steps ({settings.max_mode_steps})")
                    return

                # Run mode - modes emit their own StatusEvents
                if mode not in self.modes:
                    logger.error("Unknown mode dispatched mode=%s; falling back to answer", mode)
                    fallback_mode = "answer"
                    if fallback_mode not in self.modes:
                        yield ErrorEvent(detail="Something went wrong. Please try again later.")
                        return
                    mode = fallback_mode
                mode_instance = self.modes[mode]
                result: Optional[ModeResult] = None
                async for event in _with_heartbeats(mode_instance.run(self, chat_history)):
                    if isinstance(event, ModeResult):
                        result = event
                    elif isinstance(event, (HeartbeatEvent, StatusEvent)):
                        yield event
                    else:
                        yield event

                if result is None:
                    raise RuntimeError(f"Mode {mode} did not yield ModeResult")

                if result.terminal:
                    meta = chat_history.metadata
                    usage = chat_history.usage_calls
                    total_input = sum(c.input_tokens for c in usage)
                    total_output = sum(c.output_tokens for c in usage)
                    logger.info(
                        "handle_message_streaming done mode_runs=%s tool_calls=%s "
                        "total_in=%d total_out=%d calls=%d "
                        "chain_id=%s summary_len=%d summary_up_to=%d",
                        meta.mode_runs.model_dump(),
                        meta.tool_calls.model_dump(),
                        total_input, total_output, len(usage),
                        meta.conversation_chain_id[:8] if meta.conversation_chain_id else None,
                        len(meta.conversation_summary) if meta.conversation_summary else 0,
                        meta.summary_up_to_index,
                    )
                    return

                if result.next_mode is None:
                    logger.info(
                        "handle_message_streaming done (no next mode) mode_runs=%s tool_calls=%s",
                        chat_history.metadata.mode_runs.model_dump(),
                        chat_history.metadata.tool_calls.model_dump(),
                    )
                    return

                if result.next_mode not in self.modes:
                    logger.error(
                        "Mode yielded unknown next_mode current=%s next=%s; falling back to answer",
                        mode,
                        result.next_mode,
                    )
                    fallback_mode = "answer"
                    if fallback_mode not in self.modes:
                        yield ErrorEvent(detail="Something went wrong. Please try again later.")
                        return
                    mode = fallback_mode
                    continue

                mode = result.next_mode
                logger.info("handle_message_streaming transition mode=%s", mode)
        except Exception:
            logger.exception("handle_message_streaming crashed mode=%s", mode)
            yield ErrorEvent(detail="Something went wrong. Please try again later.")
            return

    async def _context_for_mode(self, chat_history: ChatHistory) -> List[dict]:
        """Build unified context: [summary] + [recent messages within token budget].

        All modes share the same context window.  The token budget
        (``settings.context_max_tokens``) determines how many recent messages
        are included.  When messages fall outside that window and are not yet
        summarised, ``_update_conversation_summary`` is called with the 80/20
        rule: summarise enough to leave only ~20% of the budget unsummarised,
        giving ~80% headroom before the next summary fires.
        """
        messages = chat_history.llm_chat_history_raw.conversation_history
        meta = chat_history.metadata
        max_tokens = settings.context_max_tokens

        # Walk backwards — fit as many recent messages as the budget allows
        recent = []
        total = 0
        for msg in reversed(messages):
            t = _count_message_tokens(msg)
            if total + t > max_tokens and recent:
                break
            recent.insert(0, msg)
            total += t

        # If messages fell outside the window that aren't yet summarised — trigger
        window_start = len(messages) - len(recent)
        if window_start > meta.summary_up_to_index:
            # 80/20 rule: keep only 20% of budget as unsummarised for headroom
            keep_tokens = max_tokens // 5
            keep = []
            keep_total = 0
            for msg in reversed(messages):
                t = _count_message_tokens(msg)
                if keep_total + t > keep_tokens and keep:
                    break
                keep.insert(0, msg)
                keep_total += t
            summarise_up_to = len(messages) - len(keep)
            await self._update_conversation_summary(chat_history, up_to=summarise_up_to)
            recent = keep

        # Build context
        items: List[dict] = []
        if meta.conversation_summary:
            items.append({
                "role": "system",
                "content": f"[Oppsummering av tidligere samtale]\n"
                           f"{meta.conversation_summary}",
            })
        items.extend({"role": m.role, "content": m.content} for m in recent)
        return items

    async def _responses(
        self,
        *,
        instructions: str,
        input_items: List[dict],
        tools: List[dict] | None = None,
        text_format: dict | None = None,
        model: str | None = None,
        tool_choice: str | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
        max_output_tokens: int | None = None,
    ):
        """Make a non-streaming OpenAI Responses API call."""
        effective_model = model or settings.openai_model
        params: dict = {
            "model": effective_model,
            "instructions": instructions,
            "input": input_items,
        }
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if text_format is not None:
            params["text"] = {"format": text_format}
        if previous_response_id is not None:
            params["previous_response_id"] = previous_response_id
        if store is not None:
            params["store"] = store
        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens

        logger.info(
            "responses.create start model=%s messages=%s tools=%s",
            effective_model,
            len(input_items),
            [t.get("name") for t in (tools or [])],
        )

        max_retries = settings.openai_max_retries
        for attempt in range(max_retries + 1):
            started = time.monotonic()
            inner_task = asyncio.ensure_future(self.openai_client.responses.create(**params))
            try:
                response = await asyncio.shield(inner_task)
                break
            except asyncio.CancelledError:
                logger.info("responses.create cancelled externally, waiting for OpenAI to complete...")
                response = await inner_task
                elapsed_ms = int((time.monotonic() - started) * 1000)
                usage = getattr(response, "usage", None)
                logger.info(
                    "responses.create completed after external cancellation id=%s ms=%s in=%s out=%s",
                    getattr(response, "id", "?"),
                    elapsed_ms,
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                )
                break
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable_openai_error(exc):
                    raise
                delay = _retry_delay_seconds(attempt, settings.openai_retry_backoff_seconds)
                logger.warning(
                    "responses.create retrying model=%s attempt=%s/%s delay=%.2fs error=%s",
                    effective_model,
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        else:  # pragma: no cover
            raise RuntimeError("responses.create retry loop exhausted without returning")

        elapsed_ms = int((time.monotonic() - started) * 1000)
        usage = getattr(response, "usage", None)
        logger.info(
            "responses.create ok id=%s ms=%s in=%s out=%s",
            getattr(response, "id", "?"),
            elapsed_ms,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )
        return response

    async def _tracked_responses(
        self,
        *,
        instructions: str,
        input_items: List[dict],
        tools: List[dict] | None = None,
        text_format: dict | None = None,
        model: str | None = None,
        tool_choice: str | None = None,
        previous_response_id: str | None = None,
        store: bool | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[Any, TrackedUsage]:
        """Like _responses but also returns TrackedUsage for telemetry."""
        effective_model = model or settings.openai_model
        response = await self._responses(
            instructions=instructions,
            input_items=input_items,
            tools=tools,
            text_format=text_format,
            model=effective_model,
            tool_choice=tool_choice,
            previous_response_id=previous_response_id,
            store=store,
            max_output_tokens=max_output_tokens,
        )

        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        tracked = TrackedUsage(
            model=effective_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        logger.info(
            "tracked_responses usage model=%s in=%d out=%d",
            effective_model, input_tokens, output_tokens,
        )
        return response, tracked

    def _record_usage(self, chat_history: ChatHistory, tracked: TrackedUsage) -> None:
        """Record token usage in chat history for telemetry."""
        chat_history.usage_calls.append(UsageCall(
            model=tracked.model,
            input_tokens=tracked.input_tokens,
            output_tokens=tracked.output_tokens,
        ))

    def _evict_stale_sources(self, chat_history: ChatHistory) -> None:
        """Remove legal sources not cited in the last N user turns."""
        source_last_used = chat_history.metadata.source_last_used
        if not source_last_used:
            return
        current_turn = _ui_user_turn_count(chat_history)
        stale_ids = {
            ref_id
            for ref_id, last_used in source_last_used.items()
            if current_turn - last_used > settings.source_stale_messages
        }
        if not stale_ids:
            return
        chat_history.metadata.retrieval.results = [
            doc for doc in chat_history.metadata.retrieval.results
            if doc["work_ref_id"] not in stale_ids
        ]
        for ref_id in stale_ids:
            del chat_history.metadata.source_last_used[ref_id]
        logger.info(
            "evicted %d stale sources: %s (current_turn=%d threshold=%d)",
            len(stale_ids), stale_ids, current_turn, settings.source_stale_messages,
        )

    def _extract_function_calls(self, response, chat_history: ChatHistory) -> List[ToolCall]:
        """Extract tool calls from OpenAI response."""
        calls: List[ToolCall] = []
        for item in response.output:
            if item.type != "function_call":
                continue
            # Increment the counter for this tool call type
            current = getattr(chat_history.metadata.tool_calls, item.name)
            setattr(chat_history.metadata.tool_calls, item.name, current + 1)
            try:
                cleaned = _strip_trailing_json(item.arguments)
                arguments = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "dropping malformed tool call name=%s call_id=%s error=%s raw=%s",
                    item.name,
                    item.call_id,
                    exc,
                    _preview(item.arguments, 200),
                )
                continue
            calls.append(ToolCall(name=item.name, call_id=item.call_id, arguments=arguments))
        return calls

    async def _call_structured_response(
        self,
        *,
        chat_history: ChatHistory,
        schema_model: type[BaseModel],
        schema_name: str,
        instructions: str,
        input_items: List[dict],
        model: str | None = None,
        store: bool | None = None,
        previous_response_id: str | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[Any, BaseModel]:
        """Run a structured non-streaming LLM call with one schema-repair retry."""
        effective_instructions = instructions
        max_retries = settings.structured_output_max_retries
        for attempt in range(max_retries + 1):
            response, tracked = await self._tracked_responses(
                instructions=effective_instructions,
                input_items=input_items,
                text_format=_text_format(schema_name, schema_model),
                model=model,
                previous_response_id=previous_response_id,
                store=store,
                max_output_tokens=max_output_tokens,
            )
            self._record_usage(chat_history, tracked)
            message_text = _extract_message_text(response)
            try:
                payload = _validate_json_model(schema_model, message_text, label=schema_name)
                return response, payload
            except StructuredOutputError as exc:
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "structured response retrying schema=%s attempt=%s/%s raw=%s error=%s",
                    schema_name,
                    attempt + 1,
                    max_retries,
                    _preview(message_text, 300),
                    exc.original,
                )
                _shared = _load_prompts("agents.shared.languages")
                effective_instructions = instructions + _shared.JSON_REPAIR_SUFFIX

    async def _update_conversation_summary(self, chat_history: ChatHistory, *, up_to: int) -> None:
        """Fold messages[summary_up_to_index:up_to] into the rolling summary.

        Called by ``_context_for_mode`` when messages overflow the token budget.
        """
        messages = chat_history.llm_chat_history_raw.conversation_history
        meta = chat_history.metadata
        new_messages = messages[meta.summary_up_to_index:up_to]
        if not new_messages:
            return

        logger.info(
            "updating conversation summary new_msgs=%d old_up_to=%d new_up_to=%d",
            len(new_messages), meta.summary_up_to_index, up_to,
        )

        _shared = _load_prompts("agents.shared.languages")
        parts: List[str] = []
        if meta.conversation_summary:
            parts.append(f"{_shared.SUMMARY_PREVIOUS_LABEL}\n{meta.conversation_summary}")
        parts.append(_shared.SUMMARY_NEW_MESSAGES_LABEL)
        for m in new_messages:
            parts.append(f"{m.role}: {m.content[:500]}")

        max_summary_tokens = settings.context_max_tokens // 4
        max_summary_words = max_summary_tokens // 3

        try:
            _, payload = await self._call_structured_response(
                chat_history=chat_history,
                schema_model=SummaryPayload,
                schema_name="summary",
                instructions=_shared.summary_instructions(max_summary_words),
                input_items=[{"role": "user", "content": "\n\n".join(parts)}],
                model=SUMMARY_MODEL,
                store=False,
            )
        except StructuredOutputError as exc:
            logger.warning("conversation summary update skipped due to invalid output: %s", exc)
            return

        summary_text = payload.summary.strip()

        raw_tokens = _count_tokens(summary_text)
        if raw_tokens > max_summary_tokens:
            enc = _get_encoder()
            summary_text = enc.decode(enc.encode(summary_text)[:max_summary_tokens])
            logger.warning("summary truncated from %d to %d tokens", raw_tokens, max_summary_tokens)

        meta.conversation_summary = summary_text
        meta.summary_up_to_index = up_to

        logger.info(
            "conversation summary updated summary_len=%d summary_up_to=%d",
            len(meta.conversation_summary),
            meta.summary_up_to_index,
        )

    def _prune_old_messages(self, chat_history: ChatHistory) -> None:
        """Storage safety valve: drop oldest messages when over hard cap.

        Rebases summary_up_to_index and invalidates the conversation chain
        (messages it references are gone).
        """
        max_messages = settings.max_stored_messages
        messages = chat_history.llm_chat_history_raw.conversation_history
        if len(messages) <= max_messages:
            return

        dropped = len(messages) - max_messages
        chat_history.llm_chat_history_raw.conversation_history = messages[dropped:]

        meta = chat_history.metadata
        meta.summary_up_to_index = max(0, meta.summary_up_to_index - dropped)

        # Chain is invalid after pruning — messages it references are gone
        meta.conversation_chain_id = None
        meta.chain_message_count = 0
        meta.chain_context_hash = None
        meta.chain_last_mode = None

        logger.info(
            "safety-valve pruned %d oldest messages, rebased summary_up_to=%d",
            dropped, meta.summary_up_to_index,
        )

