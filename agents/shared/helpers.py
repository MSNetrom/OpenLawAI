"""Language-agnostic shared utilities for the legal assistant."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import suppress
from typing import Any, AsyncIterator, Dict, List, TypeVar, Union

import tiktoken
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from agents.locale import load_prompts
from agents.models import (
    ChatMetadata,
    ChatHistory,
    HeartbeatEvent,
    HEARTBEAT_INTERVAL_SECONDS,
    ModeName,
    SetModeArgs,
    settings,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _get_prompts():
    return load_prompts("agents.shared.languages")


class StructuredOutputError(ValueError):
    """Raised when an LLM/tool response does not match the expected schema."""

    def __init__(self, *, label: str, raw_output: str, original: Exception) -> None:
        self.label = label
        self.raw_output = raw_output
        self.original = original
        super().__init__(f"Invalid structured output for {label}: {original}")


# --- JSON Schema Helpers ---

def _text_format(name: str, model: type[BaseModel]) -> dict:
    return {"type": "json_schema", "name": name, "strict": True, "schema": model.model_json_schema()}


def _tool_parameters(model: type[BaseModel]) -> dict:
    return model.model_json_schema()


def _ui_user_turn_count(chat_history: ChatHistory) -> int:
    """Return the number of user turns in the visible conversation history."""
    return sum(
        1
        for message in chat_history.ui_chat_history_raw.conversation_history
        if message.role == "user"
    )


def _set_mode_tool(allowed: List[ModeName]) -> dict:
    """Build set_mode tool definition for OpenAI."""
    prompts = _get_prompts()
    schema = _tool_parameters(SetModeArgs)
    schema["properties"]["mode"]["enum"] = allowed

    return {
        "type": "function",
        "name": "set_mode",
        "description": prompts.SET_MODE_DESCRIPTION_PREFIX + "\n"
        + "\n".join([f"- {mode}: {prompts.MODE_GUIDANCE[mode]}" for mode in allowed if mode in prompts.MODE_GUIDANCE]),
        "parameters": schema,
    }


# --- Text Utilities ---

def _preview(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_message_text(response) -> str:
    """Extract only the 'message' output text from a response, ignoring reasoning etc."""
    for item in response.output:
        if item.type == "message":
            for block in getattr(item, "content", []):
                if hasattr(block, "text"):
                    return block.text
    return response.output_text


def _extract_message_from_streaming_json(raw_buffer: str, field: str = "answer") -> str:
    """Extract text from partial JSON using json_repair library."""
    try:
        parsed = repair_json(raw_buffer, return_objects=True, stream_stable=True)
        if isinstance(parsed, dict):
            return parsed.get(field, "")
        return ""
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _strip_trailing_json(raw: str) -> str:
    """Extract the first complete JSON value, discarding any trailing characters."""
    decoded, _ = json.JSONDecoder().raw_decode(raw.strip())
    return json.dumps(decoded, ensure_ascii=False)


def _validate_json_model(model: type[BaseModel], raw_output: str, *, label: str) -> BaseModel:
    """Validate JSON output and raise a structured error with context on failure."""
    try:
        cleaned = _strip_trailing_json(raw_output)
        return model.model_validate_json(cleaned)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(label=label, raw_output=raw_output, original=exc) from exc


def _validate_model_data(model: type[BaseModel], data: Any, *, label: str) -> BaseModel:
    """Validate already-decoded data and raise a structured error on failure."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            label=label,
            raw_output=json.dumps(data, ensure_ascii=False, default=str),
            original=exc,
        ) from exc


def _trim_documents_for_context(
    documents: List[Dict[str, Any]],
    max_total_docs: int,
    max_chunks_per_doc: int,
) -> List[Dict[str, Any]]:
    """Trim documents to fit context budget."""
    trimmed: List[Dict[str, Any]] = []
    docs_by_score = sorted(documents, key=lambda d: d["score"], reverse=True)
    for doc in docs_by_score:
        if len(trimmed) >= max_total_docs:
            break
        chunks = sorted(doc["chunks"], key=lambda c: c["score"], reverse=True)
        trimmed_doc = dict(doc)
        trimmed_doc["chunks"] = chunks[:max_chunks_per_doc]
        trimmed.append(trimmed_doc)

    trimmed.sort(key=lambda d: d["score"], reverse=True)
    return trimmed


def _doc_titles_for_context(documents: List[Dict[str, Any]]) -> str:
    """Compact document titles for context — locale-aware."""
    prompts = _get_prompts()
    return prompts.doc_titles_for_context(documents)


def _context_hash(meta: ChatMetadata) -> str:
    """Sorted, stable hash over all chain-relevant context."""
    blob = json.dumps({
        "retrieval": sorted(
            (d["work_ref_id"], sorted(c["chunk_ref_id"] for c in d["chunks"]))
            for d in meta.retrieval.results
        ),
        "user_docs": sorted(
            (d.id, sorted(c.chunk_index for c in d.chunks))
            for d in meta.user_docs.documents if d.chunks
        ),
        "generated": sorted(
            (d.id, d.filename, d.announced)
            for d in meta.generated_documents
        ),
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --- Token Counting ---

_encoder: tiktoken.Encoding | None = None
_encoder_model: str | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder, _encoder_model
    model_name = settings.openai_model
    if _encoder is None or _encoder_model != model_name:
        try:
            _encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            _encoder = tiktoken.encoding_for_model("gpt-4o")
        _encoder_model = model_name
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _count_message_tokens(msg) -> int:
    """Token count for a single Message (content + role overhead)."""
    return _count_tokens(msg.content) + 4


# --- Heartbeat Wrapper ---

async def _with_heartbeats(async_iter: AsyncIterator[T], interval: float = HEARTBEAT_INTERVAL_SECONDS) -> AsyncIterator[Union[T, HeartbeatEvent]]:
    """Wrap an async iterator to interleave heartbeat events during long waits."""
    next_task: asyncio.Task[T] | None = None
    try:
        while True:
            next_task = asyncio.create_task(async_iter.__anext__())
            while True:
                done, _ = await asyncio.wait({next_task}, timeout=interval)
                if done:
                    try:
                        yield next_task.result()
                    except StopAsyncIteration:
                        return
                    next_task = None
                    break
                yield HeartbeatEvent()
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            with suppress(asyncio.CancelledError):
                await next_task
        aclose = getattr(async_iter, "aclose", None)
        if aclose is not None:
            await aclose()
