"""AnswerMode - Streams the final answer to the user using OpenAI Responses API."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Union
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from config.model_routing import MODE_MODELS

from agents.locale import load_prompts
from agents.mode_base import Mode
from agents.models import (
    AnswerPayload,
    ChatHistory,
    ChunkEvent,
    ErrorEvent,
    HEARTBEAT_INTERVAL_SECONDS,
    HeartbeatEvent,
    ModeName,
    ModeResult,
    QualityModeName,
    StatusEvent,
    StreamEvent,
    TrackedUsage,
    settings,
)
from agents.shared import (
    _context_hash,
    _extract_message_from_streaming_json,
    _preview,
    _strip_trailing_json,
    _text_format,
    _ui_user_turn_count,
)

if TYPE_CHECKING:
    from chat_manager import ChatManager

_prompts = load_prompts("agents.answer.languages")

logger = logging.getLogger(__name__)


def _is_retryable_stream_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def _stream_retry_delay_seconds(attempt: int) -> float:
    return settings.openai_retry_backoff_seconds * (attempt + 1)


def _persist_failed_answer(chat_history: ChatHistory, answer: str) -> None:
    chat_history.ui_chat_history_raw.new_message(role="assistant", content=answer)
    chat_history.llm_chat_history_raw.new_message(role="assistant", content=answer)
    meta = chat_history.metadata
    meta.conversation_chain_id = None
    meta.chain_message_count = 0
    meta.chain_context_hash = None
    meta.chain_last_mode = None
    meta.chain_reused_last = False


class AnswerMode(Mode):
    """Streams the final answer to the user using OpenAI Responses API."""
    name: ModeName = "answer"
    models: Dict[QualityModeName, str] = MODE_MODELS["answer"]

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.answer += 1
        
        yield StatusEvent(message=_prompts.STATUS_FORMULATING, mode=self.name)
        
        quality_mode: QualityModeName = chat_history.metadata.quality_mode
        answer_model = self.models[quality_mode]
        
        logger.info("mode=answer streaming start messages=%s model=%s", len(chat_history.llm_chat_history_raw.conversation_history), answer_model)
        
        retrieval = chat_history.metadata.retrieval
        documents = retrieval.results
        
        # Include user document results if available
        user_doc_results = [
            doc.for_llm()
            for doc in chat_history.metadata.user_docs.documents
            if doc.chunks
        ]

        # Include generated document references
        all_generated_docs = chat_history.metadata.generated_documents
        unannounced_docs = [doc for doc in all_generated_docs if not doc.announced]

        # Build static instructions (cacheable) and document context (input_items prefix)
        instructions = _prompts.build_static_instructions(
            has_user_docs=bool(user_doc_results),
            has_unannounced_docs=bool(unannounced_docs),
        )
        doc_context_msg = _prompts.build_document_context_message(
            documents,
            user_doc_results if user_doc_results else None,
            all_generated_docs if all_generated_docs else None,
        )

        # --- Chain logic ---
        # AnswerMode only reuses chains it created (not ClarifyMode's), because
        # Clarify's chain lacks full legal-doc content needed for citations.
        meta = chat_history.metadata
        current_hash = _context_hash(meta)
        chain_valid = (
            meta.conversation_chain_id is not None
            and meta.chain_context_hash == current_hash
            and meta.chain_last_mode == "answer"
            and meta.chain_message_count < len(chat_history.llm_chat_history_raw.conversation_history)
        )

        meta.chain_reused_last = chain_valid
        if chain_valid:
            # Chain intact — only send new messages since chain was set
            all_messages = chat_history.llm_chat_history_raw.conversation_history
            new_messages = all_messages[meta.chain_message_count:]
            input_items = [{"role": m.role, "content": m.content} for m in new_messages]
            logger.info("mode=answer chain valid chain_id=%s new_msgs=%d", meta.conversation_chain_id, len(input_items))
        else:
            # Chain broken — full bounded context
            conversation_context = await manager._context_for_mode(chat_history)
            input_items = [doc_context_msg, *conversation_context]
            logger.info("mode=answer chain broken, sending full context msgs=%d", len(input_items))

        started = time.monotonic()
        chunk_count = 0
        raw_buffer = ""
        last_yielded_length = 0
        extracted_answer = ""
        captured_usage: Dict[str, Any] = {}
        stream_error: Exception | None = None
        elapsed_ms = 0

        params: dict = {
            "model": answer_model,
            "instructions": instructions,
            "input": input_items,
            "text": {"format": _text_format("answer", AnswerPayload)},
            "stream": True,
            "store": True,
        }
        if chain_valid:
            params["previous_response_id"] = meta.conversation_chain_id

        logger.info("responses.create streaming start model=%s messages=%s", answer_model, len(params["input"]))

        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def stream_to_queue():
            nonlocal captured_usage, stream_error
            max_attempts = max(1, settings.openai_max_retries + 1)
            try:
                for attempt in range(max_attempts):
                    completed_received = False
                    saw_delta = False
                    try:
                        async with await manager.openai_client.responses.create(**params) as stream:
                            async for event in stream:
                                event_type = getattr(event, "type", None)
                                if event_type == "response.output_text.delta":
                                    delta = getattr(event, "delta", "")
                                    if delta:
                                        saw_delta = True
                                        await chunk_queue.put(delta)
                                elif event_type == "response.completed":
                                    response = event.response
                                    elapsed_ms = int((time.monotonic() - started) * 1000)
                                    usage = response.usage
                                    captured_usage["input_tokens"] = usage.input_tokens
                                    captured_usage["output_tokens"] = usage.output_tokens
                                    captured_usage["response_id"] = response.id
                                    completed_received = True
                                    logger.info(
                                        "responses.create streaming ok id=%s ms=%s in=%s out=%s chunks=%s",
                                        getattr(response, "id", "?"), elapsed_ms,
                                        usage.input_tokens, usage.output_tokens, chunk_count,
                                    )
                        if not completed_received:
                            raise RuntimeError("responses.create stream ended without response.completed")
                        return
                    except Exception as exc:
                        if (
                            attempt + 1 < max_attempts
                            and not saw_delta
                            and chunk_count == 0
                            and _is_retryable_stream_error(exc)
                        ):
                            delay = _stream_retry_delay_seconds(attempt)
                            logger.warning(
                                "responses.create streaming retrying model=%s attempt=%s/%s delay=%.2fs error=%s",
                                answer_model,
                                attempt + 1,
                                max_attempts,
                                delay,
                                exc,
                            )
                            await asyncio.sleep(delay)
                            continue
                        stream_error = exc
                        return
            finally:
                await chunk_queue.put(None)

        stream_task = asyncio.create_task(stream_to_queue())
        heartbeat_interval = HEARTBEAT_INTERVAL_SECONDS
        last_progress_at = started

        try:
            while True:
                try:
                    text = await asyncio.wait_for(chunk_queue.get(), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - started >= settings.stream_total_timeout_seconds:
                        stream_error = RuntimeError(
                            f"Answer stream exceeded total timeout ({settings.stream_total_timeout_seconds}s)"
                        )
                        break
                    if now - last_progress_at >= settings.stream_idle_timeout_seconds:
                        stream_error = RuntimeError(
                            f"Answer stream exceeded idle timeout ({settings.stream_idle_timeout_seconds}s)"
                        )
                        break
                    yield HeartbeatEvent()
                    continue

                if text is None:
                    break

                raw_buffer += text
                chunk_count += 1
                last_progress_at = time.monotonic()
                extracted_text = _extract_message_from_streaming_json(raw_buffer, "answer")
                extracted_answer = extracted_text
                if len(extracted_text) > last_yielded_length:
                    new_text = extracted_text[last_yielded_length:]
                    last_yielded_length = len(extracted_text)
                    if chunk_count <= 3 or chunk_count % 50 == 0:
                        logger.info("streaming chunk %d: yielding %d chars", chunk_count, len(new_text))
                    yield ChunkEvent(text=new_text)
            elapsed_ms = int((time.monotonic() - started) * 1000)
        finally:
            if not stream_task.done():
                stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream_task
            else:
                await stream_task

        if stream_error is not None:
            if extracted_answer:
                _persist_failed_answer(chat_history, extracted_answer)
            logger.error("Answer stream failed: %s raw=%s", stream_error, _preview(raw_buffer, 500))
            yield ErrorEvent(detail=f"Answer stream failed: {stream_error}")
            yield ModeResult(next_mode=None, terminal=True)
            return

        try:
            cleaned = _strip_trailing_json(raw_buffer)
            payload = AnswerPayload.model_validate_json(cleaned)
        except Exception as e:
            if extracted_answer:
                _persist_failed_answer(chat_history, extracted_answer)
            if captured_usage:
                response_id = captured_usage["response_id"]
                manager._record_usage(chat_history, TrackedUsage(
                    model=answer_model,
                    input_tokens=captured_usage["input_tokens"],
                    output_tokens=captured_usage["output_tokens"],
                ))
                logger.info(
                    "mode=answer persisted partial response after parse failure id=%s chars=%s",
                    response_id,
                    len(extracted_answer),
                )
            logger.error("Failed to parse answer response: %s raw=%s", e, _preview(raw_buffer, 500))
            yield ErrorEvent(detail=f"Failed to parse response: {e}")
            yield ModeResult(next_mode=None, terminal=True)
            return
        answer = payload.answer
        logger.info("mode=answer streaming done ms=%s chunks=%s chars=%s", elapsed_ms, chunk_count, len(answer))

        response_id = captured_usage["response_id"]
        manager._record_usage(chat_history, TrackedUsage(
            model=answer_model,
            input_tokens=captured_usage["input_tokens"],
            output_tokens=captured_usage["output_tokens"],
        ))

        # Update chat history
        chat_history.ui_chat_history_raw.new_message(role="assistant", content=answer)
        chat_history.llm_chat_history_raw.new_message(role="assistant", content=answer)

        # Track which sources were used
        current_turn = _ui_user_turn_count(chat_history)
        for ref_id in payload.used_source_ids:
            chat_history.metadata.source_last_used[ref_id] = current_turn
        logger.info("mode=answer used_source_ids=%s", payload.used_source_ids)

        # Mark documents as announced
        for doc in unannounced_docs:
            doc.announced = True

        # Update chain metadata for next turn (after announcing, so hash reflects final state)
        meta.conversation_chain_id = response_id
        meta.chain_message_count = len(chat_history.llm_chat_history_raw.conversation_history)
        meta.chain_context_hash = _context_hash(meta)
        meta.chain_last_mode = "answer"

        yield ModeResult(next_mode=None, terminal=True)
