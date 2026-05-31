"""DecideMode - Evaluates the current state and decides the next action using tool calling."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Union

from config.model_routing import MODE_MODELS
from agents.mode_base import Mode
from agents.models import (
    ChatHistory,
    ModeName,
    ModeResult,
    QualityModeName,
    SetModeArgs,
    StatusEvent,
    StreamEvent,
    UserDoc,
    UserDocsState,
    settings,
)
from agents.shared import StructuredOutputError, _set_mode_tool, _validate_model_data
from agents.locale import load_prompts

if TYPE_CHECKING:
    from chat_manager import ChatManager

logger = logging.getLogger(__name__)

_prompts = load_prompts("agents.decide.languages")


def _get_allowed_modes(
    documents: List[Dict],
    user_docs: UserDocsState,
    retrieval_calls: int,
    max_retrieval_passes: int,
) -> List[ModeName]:
    """Determine which modes are allowed based on context."""
    allowed: List[ModeName] = ["answer"]

    unretrieved_docs = [d for d in user_docs.documents if d.status == "ready" and not d.retrieved]
    if unretrieved_docs:
        allowed.append("user_doc_retrieve")

    if retrieval_calls < max_retrieval_passes:
        allowed.append("retrieve")

    allowed.append("generate")
    allowed.append("clarify")

    return allowed


class DecideMode(Mode):
    """Evaluates the current state and decides the next action using tool calling."""
    name: ModeName = "decide"
    models: Dict[QualityModeName, str] = MODE_MODELS["decide"]

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.decide += 1
        logger.info("mode=decide start messages=%s", len(chat_history.llm_chat_history_raw.conversation_history))
        
        user_docs = chat_history.metadata.user_docs
        
        # Check for pending documents FIRST - process them before any LLM calls
        pending_docs = [d for d in user_docs.documents if d.status == "pending"]
        if pending_docs:
            logger.info("mode=decide: %d pending documents detected, going to process_documents", len(pending_docs))
            yield ModeResult(next_mode="process_documents")
            return
        
        # Force user_doc_retrieve when ready docs exist but haven't been retrieved
        ready_docs = [d for d in user_docs.documents if d.status == "ready"]
        unretrieved_docs = [d for d in ready_docs if not d.retrieved]
        if unretrieved_docs:
            logger.info("mode=decide: %d ready docs not retrieved, forcing user_doc_retrieve", len(unretrieved_docs))
            yield ModeResult(next_mode="user_doc_retrieve")
            return
        
        # Fast mode: skip LLM call, route directly to answer
        quality_mode: QualityModeName = chat_history.metadata.quality_mode
        if quality_mode == "fast":
            logger.info("mode=decide fast-path, skipping LLM, routing to answer")
            yield ModeResult(next_mode="answer")
            return

        yield StatusEvent(message=_prompts.STATUS_ANALYSING, mode=self.name)
        
        # Build context for decide agent
        documents = chat_history.metadata.retrieval.results
        retrieval_calls: int = chat_history.metadata.retrieval_calls
        
        allowed_modes = _get_allowed_modes(
            documents=documents,
            user_docs=user_docs,
            retrieval_calls=retrieval_calls,
            max_retrieval_passes=settings.max_retrieval_passes,
        )

        try:
            system_prompt = _prompts.build_system_prompt(
                documents=documents,
                user_docs=user_docs,
                retrieval_calls=retrieval_calls,
                max_retrieval_passes=settings.max_retrieval_passes,
                retrieval_coverage=chat_history.metadata.retrieval_coverage,
                allowed_modes=allowed_modes,
            )

            model = self.models[quality_mode]

            # Use tool calling with set_mode tool
            tools = [_set_mode_tool(allowed_modes)]
            context = await manager._context_for_mode(chat_history)
            response, tracked = await manager._tracked_responses(
                instructions=system_prompt,
                input_items=context,
                tools=tools,
                model=model,
                tool_choice="required",
                store=False,
                max_output_tokens=None,
            )
            manager._record_usage(chat_history, tracked)
            logger.info(
                "mode=decide response ready output_items=%s in=%s out=%s allowed=%s",
                len(response.output),
                tracked.input_tokens,
                tracked.output_tokens,
                allowed_modes,
            )

            # Extract the tool call
            calls = manager._extract_function_calls(response, chat_history)
            if calls:
                if len(calls) > 1:
                    logger.warning("mode=decide got %d set_mode calls, using first", len(calls))
                call = calls[0]
                if call.name != "set_mode":
                    raise RuntimeError(f"Expected set_mode, got {call.name}")
                try:
                    args = _validate_model_data(SetModeArgs, call.arguments, label="set_mode")
                except StructuredOutputError as exc:
                    logger.warning("mode=decide invalid set_mode args, defaulting to answer: %s", exc)
                    yield ModeResult(next_mode="answer")
                    return
                next_mode = args.mode
                if next_mode not in allowed_modes:
                    logger.warning("mode=decide got disallowed mode %s, defaulting to answer", next_mode)
                    yield ModeResult(next_mode="answer")
                    return
                logger.info("mode=decide -> %s", next_mode)
                yield ModeResult(next_mode=next_mode)
                return

            # No tool call - shouldn't happen with tool_choice="required", but default to answer
            logger.warning("mode=decide: no tool call received, defaulting to answer")
            yield ModeResult(next_mode="answer")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mode=decide failed, defaulting to answer")
            yield ModeResult(next_mode="answer")
