"""ProcessDocumentsMode - Extracts text from pending user documents.

Weaviate ingestion is deferred to UserDocRetrieveMode (lazy ingestion).
Small documents that fit within the chunk budget skip Weaviate entirely.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncIterator, Union

import httpx

from chatdb.models import UserDocument
from config.app_settings import upload_settings
from legal_pipeline.document_extractor import get_extractor
from legal_pipeline.ingestor import chunk_text, count_tokens

from agents.mode_base import Mode
from agents.models import (
    ChatHistory,
    ErrorEvent,
    ModeName,
    ModeResult,
    StatusEvent,
    StreamEvent,
    UserDoc,
)

if TYPE_CHECKING:
    from chat_manager import ChatManager

logger = logging.getLogger(__name__)


def _is_systemic_extraction_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code in {408, 429} or status_code >= 500
    return False


class ProcessDocumentsMode(Mode):
    """Extracts text from pending user documents.

    Weaviate ingestion is deferred to UserDocRetrieveMode (lazy).
    This runs during the chat flow so heartbeats keep the connection alive.
    Documents are extracted in a thread pool to not block the async loop.
    """

    name: ModeName = "process_documents"

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.process_documents += 1
        logger.info("mode=process_documents start")

        conversation_id = chat_history.metadata.conversation_id
        if not conversation_id:
            logger.warning("mode=process_documents: no conversation_id, skipping")
            yield ModeResult(next_mode="decide")
            return

        try:
            # Find pending documents for this conversation
            pending_docs = [
                doc async for doc in UserDocument.objects.filter(
                    conversation_id=conversation_id,
                    status__in=[UserDocument.Status.PENDING, UserDocument.Status.PROCESSING],
                ).aiterator()
            ]

            if not pending_docs:
                logger.info("mode=process_documents: no pending documents")
                yield ModeResult(next_mode="decide")
                return

            yield StatusEvent(message="Behandler opplastede dokumenter...", mode=self.name)

            extractor = get_extractor()
            docs_by_id = {doc.id: doc for doc in chat_history.metadata.user_docs.documents}

            for doc in pending_docs:
                doc_id = str(doc.id)
                if doc_id not in docs_by_id:
                    # Uploaded after _build_history snapshot — add to metadata
                    new_entry = UserDoc(id=doc_id, filename=doc.filename, status=doc.status)
                    chat_history.metadata.user_docs.documents.append(new_entry)
                    docs_by_id[doc_id] = new_entry
                doc_state = docs_by_id[doc_id]
                doc.status = UserDocument.Status.PROCESSING
                await doc.asave(update_fields=["status"])
                doc_state.status = doc.status

                try:
                    # Extract text in thread pool (non-blocking)
                    logger.info("mode=process_documents extracting doc=%s filename=%s", doc.id, doc.filename)
                    extracted_text = await asyncio.to_thread(
                        extractor.extract, bytes(doc.file_data), doc.filename
                    )

                    if not extracted_text.strip():
                        logger.warning("mode=process_documents: no text extracted from %s", doc.filename)
                        doc.status = UserDocument.Status.FAILED
                        await doc.asave(update_fields=["status"])
                        doc_state.status = doc.status
                        continue

                    # Compute stats locally (no Weaviate ingestion — deferred to retrieval)
                    was_truncated = len(extracted_text) > upload_settings.max_extracted_chars
                    if was_truncated:
                        extracted_text = extracted_text[:upload_settings.max_extracted_chars]

                    token_count = count_tokens(extracted_text)
                    chunk_count = len(chunk_text(extracted_text))

                    # Update document record
                    doc.extracted_text = extracted_text
                    doc.token_count = token_count
                    doc.chunk_count = chunk_count
                    doc.status = UserDocument.Status.READY
                    doc.weaviate_ingested = False
                    doc.file_data = None  # Clear raw bytes to save space
                    doc.metadata["was_truncated"] = was_truncated
                    await doc.asave(update_fields=[
                        "extracted_text",
                        "token_count",
                        "chunk_count",
                        "status",
                        "weaviate_ingested",
                        "file_data",
                        "metadata",
                    ])
                    doc_state.status = doc.status
                    doc_state.token_count = doc.token_count
                    doc_state.chunk_count = doc.chunk_count

                    logger.info(
                        "mode=process_documents done doc=%s tokens=%s chunks=%s",
                        doc.id, token_count, chunk_count,
                    )

                except Exception as e:
                    logger.exception("mode=process_documents failed doc=%s: %s", doc.id, e)
                    if _is_systemic_extraction_error(e):
                        doc.status = UserDocument.Status.PENDING
                        await doc.asave(update_fields=["status"])
                        doc_state.status = doc.status
                        raise
                    doc.status = UserDocument.Status.FAILED
                    await doc.asave(update_fields=["status"])
                    doc_state.status = doc.status

            logger.info("mode=process_documents complete, processed %d documents", len(pending_docs))
            yield ModeResult(next_mode="decide")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mode=process_documents failed, falling back to answer")
            yield ErrorEvent(detail="Kunne ikke behandle dokumentene nå. Fortsetter uten dokumentkontekst.")
            yield ModeResult(next_mode="answer")
