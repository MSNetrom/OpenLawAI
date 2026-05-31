"""Shared utilities and locale-loaded prompts for the legal assistant."""
from agents.shared.helpers import (  # noqa: F401
    StructuredOutputError,
    _context_hash,
    _count_message_tokens,
    _count_tokens,
    _doc_titles_for_context,
    _extract_message_from_streaming_json,
    _extract_message_text,
    _get_encoder,
    _get_prompts,
    _preview,
    _set_mode_tool,
    _strip_trailing_json,
    _text_format,
    _tool_parameters,
    _trim_documents_for_context,
    _ui_user_turn_count,
    _validate_json_model,
    _validate_model_data,
    _with_heartbeats,
)


def __getattr__(name: str):
    """Lazy attribute access for locale-dependent constants like SYSTEM_IDENTITY."""
    if name == "SYSTEM_IDENTITY":
        return _get_prompts().SYSTEM_IDENTITY
    raise AttributeError(f"module 'agents.shared' has no attribute {name!r}")
