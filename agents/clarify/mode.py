"""ClarifyMode - Generates clarifying questions using OpenAI Responses API streaming."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Union

from config.model_routing import MODE_MODELS

from agents.mode_base import Mode
from agents.models import (
    ChatHistory,
    ChunkEvent,
    ClarificationPayload,
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

logger = logging.getLogger(__name__)

from agents.locale import load_prompts

_prompts = load_prompts("agents.clarify.languages")


def _persist_failed_clarification(chat_history: ChatHistory, message: str) -> None:
    chat_history.ui_chat_history_raw.new_message(role="assistant", content=message)
    chat_history.llm_chat_history_raw.new_message(role="assistant", content=message)
    meta = chat_history.metadata
    meta.conversation_chain_id = None
    meta.chain_message_count = 0
    meta.chain_context_hash = None
    meta.chain_last_mode = None
    meta.chain_reused_last = False


class ClarifyMode(Mode):
    """Generates clarifying questions using OpenAI Responses API streaming."""
    name: ModeName = "clarify"
    models: Dict[QualityModeName, str] = MODE_MODELS["clarify"]

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.clarify += 1
        
        yield StatusEvent(message=_prompts.STATUS_PREPARING, mode=self.name)
        
        quality_mode: QualityModeName = chat_history.metadata.quality_mode
        clarify_model = self.models[quality_mode]
        
        logger.info("mode=clarify start messages=%s model=%s", len(chat_history.llm_chat_history_raw.conversation_history), clarify_model)
        
        documents = chat_history.metadata.retrieval.results
        user_doc_summaries = [
            f"{doc.filename}: {doc.summary}" if doc.summary else doc.filename
            for doc in chat_history.metadata.user_docs.documents
            if doc.chunks
        ]

        instructions = _prompts.build_system_prompt(documents, user_doc_summaries)

        # --- Chain logic ---
        meta = chat_history.metadata
        current_hash = _context_hash(meta)
        chain_valid = (
            meta.conversation_chain_id is not None
            and meta.chain_context_hash == current_hash
            and meta.chain_message_count < len(chat_history.llm_chat_history_raw.conversation_history)
        )

        meta.chain_reused_last = chain_valid
        if chain_valid:
            all_messages = chat_history.llm_chat_history_raw.conversation_history
            new_messages = all_messages[meta.chain_message_count:]
            input_items = [{"role": m.role, "content": m.content} for m in new_messages]
            logger.info("mode=clarify chain valid chain_id=%s new_msgs=%d", meta.conversation_chain_id, len(input_items))
        else:
            input_items = await manager._context_for_mode(chat_history)
            logger.info("mode=clarify chain broken, sending full context msgs=%d", len(input_items))

        started = time.monotonic()
        params: dict = {
            "model": clarify_model,
            "instructions": instructions,
            "input": input_items,
            "text": {"format": _text_format("clarification", ClarificationPayload)},
            "stream": True,
            "store": True,
        }
        if chain_valid:
            params["previous_response_id"] = meta.conversation_chain_id

        logger.info("responses.create clarify streaming start model=%s messages=%s", clarify_model, len(params["input"]))

        raw_buffer = ""
        last_yielded_length = 0
        chunk_count = 0
        extracted_message = ""
        captured_usage: Dict[str, Any] = {}
        stream_error: Exception | None = None

        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def shielded_consume():
            nonlocal chunk_count, captured_usage, stream_error
            completed_received = False
            try:
                async with await manager.openai_client.responses.create(**params) as stream:
                    async for event in stream:
                        event_type = getattr(event, "type", None)
                        if event_type == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                await chunk_queue.put(delta)
                        elif event_type == "response.completed":
                            response = getattr(event, "response", None)
                            if response is None:
                                raise RuntimeError("responses.create clarify completed event missing response")
                            elapsed_ms = int((time.monotonic() - started) * 1000)
                            usage = response.usage
                            captured_usage = {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "response_id": response.id,
                            }
                            completed_received = True
                            logger.info(
                                "responses.create clarify streaming ok id=%s ms=%s in=%s out=%s chunks=%s",
                                getattr(response, "id", "?"), elapsed_ms,
                                usage.input_tokens,
                                usage.output_tokens, chunk_count,
                            )
                if not completed_received:
                    raise RuntimeError("responses.create clarify stream ended without response.completed")
            except Exception as exc:
                stream_error = exc
            finally:
                await chunk_queue.put(None)

        consumer_task = asyncio.create_task(shielded_consume())
        heartbeat_interval = HEARTBEAT_INTERVAL_SECONDS
        last_progress_at = started

        try:
            while True:
                try:
                    delta = await asyncio.wait_for(chunk_queue.get(), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - started >= settings.stream_total_timeout_seconds:
                        stream_error = RuntimeError(
                            f"Clarify stream exceeded total timeout ({settings.stream_total_timeout_seconds}s)"
                        )
                        break
                    if now - last_progress_at >= settings.stream_idle_timeout_seconds:
                        stream_error = RuntimeError(
                            f"Clarify stream exceeded idle timeout ({settings.stream_idle_timeout_seconds}s)"
                        )
                        break
                    yield HeartbeatEvent()
                    continue

                if delta is None:
                    break
                raw_buffer += delta
                chunk_count += 1
                last_progress_at = time.monotonic()
                extracted_text = _extract_message_from_streaming_json(raw_buffer, "message")
                extracted_message = extracted_text
                if len(extracted_text) > last_yielded_length:
                    new_text = extracted_text[last_yielded_length:]
                    last_yielded_length = len(extracted_text)
                    yield ChunkEvent(text=new_text)
        finally:
            if not consumer_task.done():
                consumer_task.cancel()
                with suppress(asyncio.CancelledError):
                    await consumer_task
            else:
                await consumer_task

        if stream_error is not None:
            if extracted_message:
                _persist_failed_clarification(chat_history, extracted_message)
            logger.error("Clarify stream failed: %s raw=%s", stream_error, _preview(raw_buffer, 500))
            yield ErrorEvent(detail=f"Clarify stream failed: {stream_error}")
            yield ModeResult(next_mode=None, terminal=True)
            return

        try:
            cleaned = _strip_trailing_json(raw_buffer)
            payload = ClarificationPayload.model_validate_json(cleaned)
        except Exception as e:
            if extracted_message:
                _persist_failed_clarification(chat_history, extracted_message)
            if captured_usage:
                manager._record_usage(chat_history, TrackedUsage(
                    model=clarify_model,
                    input_tokens=captured_usage["input_tokens"],
                    output_tokens=captured_usage["output_tokens"],
                ))
                logger.info(
                    "mode=clarify persisted partial response after parse failure id=%s chars=%s",
                    captured_usage["response_id"],
                    len(extracted_message),
                )
            logger.error("Failed to parse clarify response: %s raw=%s", e, _preview(raw_buffer, 500))
            yield ErrorEvent(detail=f"Failed to parse response: {e}")
            yield ModeResult(next_mode=None, terminal=True)
            return

        response_id = captured_usage["response_id"]
        manager._record_usage(chat_history, TrackedUsage(
            model=clarify_model,
            input_tokens=captured_usage["input_tokens"],
            output_tokens=captured_usage["output_tokens"],
        ))

        message = payload.message
        chat_history.ui_chat_history_raw.new_message(role="assistant", content=message)
        chat_history.llm_chat_history_raw.new_message(role="assistant", content=message)

        current_turn = _ui_user_turn_count(chat_history)
        for ref_id in payload.used_source_ids:
            chat_history.metadata.source_last_used[ref_id] = current_turn
        logger.info("mode=clarify used_source_ids=%s", payload.used_source_ids)

        meta.conversation_chain_id = response_id
        meta.chain_message_count = len(chat_history.llm_chat_history_raw.conversation_history)
        meta.chain_context_hash = _context_hash(meta)
        meta.chain_last_mode = "clarify"

        logger.info("mode=clarify streaming done")
        yield ModeResult(next_mode=None, terminal=True)
