"""GenerateMode - Generates a document (PDF/DOCX) based on the conversation context using OpenAI Responses API."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator, Dict, Union

from config.model_routing import MODE_MODELS
from config.app_settings import upload_settings
from agents.mode_base import Mode
from agents.models import (
    ChatHistory,
    ErrorEvent,
    GeneratedDoc,
    GenerateDocumentPayload,
    ModeName,
    ModeResult,
    QualityModeName,
    StatusEvent,
    StreamEvent,
    settings,
)
from agents.shared import StructuredOutputError
from agents.locale import load_prompts

if TYPE_CHECKING:
    from chat_manager import ChatManager

logger = logging.getLogger(__name__)

_prompts = load_prompts("agents.generate.languages")


class GenerateMode(Mode):
    """Generates a document (PDF/DOCX) based on the conversation context using OpenAI Responses API."""

    name: ModeName = "generate"
    models: Dict[QualityModeName, str] = MODE_MODELS["generate"]

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.generate += 1
        logger.info("mode=generate start messages=%s", len(chat_history.llm_chat_history_raw.conversation_history))

        conversation_id = chat_history.metadata.conversation_id
        if not conversation_id:
            logger.warning("mode=generate: no conversation_id in metadata, cannot save document")
            yield ModeResult(next_mode="answer")
            return

        yield StatusEvent(message=_prompts.STATUS_GENERATING, mode=self.name)

        try:
            documents = chat_history.metadata.retrieval.results

            user_doc_results = [
                doc.for_llm()
                for doc in chat_history.metadata.user_docs.documents
                if doc.chunks
            ]

            quality_mode: QualityModeName = chat_history.metadata.quality_mode
            generate_model = self.models[quality_mode]

            output_format = self._detect_format_from_messages(chat_history)

            doc_context_msg = _prompts.build_document_context_message(
                documents,
                user_doc_results if user_doc_results else None,
            )
            conversation_context = await manager._context_for_mode(chat_history)
            input_items = [doc_context_msg, *conversation_context]

            _, payload = await manager._call_structured_response(
                chat_history=chat_history,
                schema_model=GenerateDocumentPayload,
                schema_name="generate_document",
                instructions=_prompts.GENERATE_STATIC_INSTRUCTIONS,
                input_items=input_items,
                model=generate_model,
                store=False,
            )
        except StructuredOutputError as exc:
            logger.error("mode=generate invalid structured output: %s", exc)
            yield ErrorEvent(detail=_prompts.ERROR_GENERATE_FAILED)
            yield ModeResult(next_mode=None, terminal=True)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mode=generate failed")
            yield ErrorEvent(detail=_prompts.ERROR_GENERATE_FAILED)
            yield ModeResult(next_mode=None, terminal=True)
            return

        markdown = payload.markdown[:upload_settings.max_generated_markdown_chars]
        title = payload.title

        from legal_pipeline.document_generator import generate_filename, markdown_to_format

        filename = generate_filename(title, output_format)
        file_bytes = await asyncio.to_thread(markdown_to_format, markdown, output_format)

        from chatdb.models import ChatConversation, GeneratedDocument

        conversation = await ChatConversation.objects.aget(pk=conversation_id)
        generated_doc = await GeneratedDocument.objects.acreate(
            conversation=conversation,
            filename=filename,
            format=output_format,
            markdown_source=markdown,
            file_data=file_bytes,
        )

        chat_history.metadata.generated_documents.append(GeneratedDoc(
            id=str(generated_doc.id),
            filename=filename,
            format=output_format,
            title=title,
        ))

        logger.info(
            "mode=generate done doc_id=%s filename=%s format=%s bytes=%s",
            generated_doc.id, filename, output_format, len(file_bytes),
        )

        yield ModeResult(next_mode="answer")

    def _detect_format_from_messages(self, chat_history: ChatHistory) -> str:
        """Detect desired output format from user messages."""
        user_messages = [
            m.content.lower()
            for m in chat_history.llm_chat_history_raw.conversation_history
            if m.role == "user"
        ]
        last_messages = " ".join(user_messages[-3:])

        if "docx" in last_messages or "word" in last_messages:
            return "docx"
        if "markdown" in last_messages or ".md" in last_messages:
            return "md"
        return "pdf"
