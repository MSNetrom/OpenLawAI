"""UserDocRetrieveMode - Henter fra brukeropplastede dokumenter med LLM-genererte søk."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, AsyncIterator, Dict, List, Union

from django.utils import timezone

from agents.locale import load_prompts
from agents.mode_base import Mode
from agents.models import (
    ChatHistory,
    ErrorEvent,
    ModeName,
    ModeResult,
    QualityModeName,
    StatusEvent,
    StreamEvent,
    SummaryPayload,
    UserDoc,
    UserDocChunk,
    UserDocQueryPayload,
    UserDocQuerySet,
    settings,
)
from agents.shared import StructuredOutputError, _preview
from chatdb.models import ChatConversation, UserDocument
from config.app_settings import search_settings
from config.model_routing import MODE_MODELS
from legal_pipeline.chunker import EmbeddingService
from legal_pipeline.ingestor import ingest_user_document
from legal_pipeline.reranker import RerankerClient
from legal_pipeline.retriever import search_user_documents
from legal_pipeline.weaviate_client import UserDocumentChunkHit, UserDocumentStore

if TYPE_CHECKING:
    from chat_manager import ChatManager

logger = logging.getLogger(__name__)

_prompts = load_prompts("agents.user_doc.languages")

RRF_K = 60


def _fuse_user_doc_rrf(
    query_results: List[List[UserDocumentChunkHit]],
    max_chunks: int,
) -> List[UserDocumentChunkHit]:
    """Chunk-level RRF fusion across multiple user doc queries.

    Each query's results are ranked by score (reranker or hybrid).
    RRF computes rank-based scores that are comparable across queries.
    Returns the top max_chunks hits by RRF score.
    """
    chunk_key = lambda h: (h.document_id, h.chunk_index)
    chunk_best: Dict[tuple, UserDocumentChunkHit] = {}
    for result_hits in query_results:
        for hit in result_hits:
            key = chunk_key(hit)
            if key not in chunk_best or hit.score > chunk_best[key].score:
                chunk_best[key] = hit

    if not chunk_best:
        return []

    query_ranks: List[Dict[tuple, int]] = []
    for result_hits in query_results:
        ranked = sorted(result_hits, key=lambda h: h.score, reverse=True)
        ranks = {chunk_key(h): rank for rank, h in enumerate(ranked, 1)}
        query_ranks.append(ranks)

    rrf_scores: Dict[tuple, float] = {}
    for key in chunk_best:
        rrf_scores[key] = sum(
            1.0 / (RRF_K + q_ranks[key])
            for q_ranks in query_ranks
            if key in q_ranks
        )

    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:max_chunks]

    hits = [chunk_best[key] for key in sorted_keys]
    logger.info(
        "user_doc_rrf queries=%d unique_chunks=%d fused=%d",
        len(query_results), len(chunk_best), len(hits),
    )
    return hits


class UserDocRetrieveMode(Mode):
    """Henter fra brukeropplastede dokumenter ved hjelp av LLM-genererte søk.

    Bruker OpenAI Responses API til å generere optimaliserte semantic- og lexical-søk
    for å finne relevante klausuler i kontrakter og andre brukerdokumenter.
    """

    name: ModeName = "user_doc_retrieve"
    models: Dict[QualityModeName, str] = MODE_MODELS["user_doc"]

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.user_doc_retrieve += 1
        logger.info("mode=user_doc_retrieve start messages=%s", len(chat_history.llm_chat_history_raw.conversation_history))

        conversation_id = chat_history.metadata.conversation_id
        if not conversation_id:
            logger.warning("mode=user_doc_retrieve: mangler conversation_id i metadata, hopper over")
            yield ModeResult(next_mode="decide")
            return

        try:
            ready_docs = [d for d in chat_history.metadata.user_docs.documents if d.status == "ready"]
            total_chunks = sum(d.chunk_count for d in ready_docs)
            budget = settings.max_user_doc_chunks

            if total_chunks <= budget:
                async for event in self._run_small_doc_path(manager, chat_history, ready_docs, conversation_id):
                    yield event
            else:
                async for event in self._run_large_doc_path(manager, chat_history, ready_docs, conversation_id):
                    yield event

            referenced_doc_ids = [d.id for d in ready_docs if d.chunks]
            if referenced_doc_ids:
                await self._update_document_references(conversation_id, referenced_doc_ids)

            yield ModeResult(next_mode="decide")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mode=user_doc_retrieve failed, falling back to answer")
            yield ErrorEvent(detail=_prompts.ERROR_USER_DOC_FAILED)
            yield ModeResult(next_mode="answer")

    async def _run_small_doc_path(
        self,
        manager: "ChatManager",
        chat_history: ChatHistory,
        ready_docs: List[UserDoc],
        conversation_id: str,
    ) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        """Use extracted_text directly when total chunks fit within budget."""
        logger.info(
            "mode=user_doc_retrieve small_doc_path docs=%d total_chunks=%d budget=%d",
            len(ready_docs), sum(d.chunk_count for d in ready_docs), settings.max_user_doc_chunks,
        )
        yield StatusEvent(message=_prompts.STATUS_READING_DOCS, mode=self.name)

        docs_by_id = {doc.id: doc for doc in ready_docs}

        yield StatusEvent(message=_prompts.STATUS_EVALUATING_DOCS, mode=self.name)

        async for db_doc in UserDocument.objects.filter(
            id__in=[d.id for d in ready_docs],
            status=UserDocument.Status.READY,
        ).only("id", "extracted_text").aiterator():
            doc = docs_by_id[str(db_doc.id)]
            doc.retrieved = True
            text = db_doc.extracted_text
            doc.chunks = [UserDocChunk(chunk_index=0, text=text, score=1.0)]
            doc.summary = await self._generate_summary(manager, chat_history, doc.chunks, doc.filename)
            logger.info(
                "mode=user_doc_retrieve small_doc doc=%s text_len=%d summary_len=%d",
                doc.filename, len(text), len(doc.summary),
            )

    async def _run_large_doc_path(
        self,
        manager: "ChatManager",
        chat_history: ChatHistory,
        ready_docs: List[UserDoc],
        conversation_id: str,
    ) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        """Lazy Weaviate ingestion + full search pipeline for large documents."""
        docs_to_ingest = [d for d in ready_docs if not d.weaviate_ingested]
        if docs_to_ingest:
            await self._lazy_ingest(docs_to_ingest, conversation_id)

        yield StatusEvent(message=_prompts.STATUS_EVALUATING_QUERY, mode=self.name)

        quality_mode: QualityModeName = chat_history.metadata.quality_mode
        model = self.models[quality_mode]
        instructions = _prompts.build_user_doc_query_prompt()

        query_context = await manager._context_for_mode(chat_history)
        try:
            _, query_set_payload = await manager._call_structured_response(
                chat_history=chat_history,
                schema_model=UserDocQuerySet,
                schema_name="user_doc_query_set",
                instructions=instructions,
                input_items=query_context,
                model=model,
                store=False,
            )
        except StructuredOutputError as exc:
            logger.warning("mode=user_doc_retrieve invalid query set, falling back to answer: %s", exc)
            yield ModeResult(next_mode="answer")
            return
        query_set = query_set_payload

        for i, qp in enumerate(query_set.queries):
            logger.info(
                "mode=user_doc_retrieve query %d/%d: semantic=%s lexical=%s",
                i + 1, len(query_set.queries),
                _preview(qp.semantic_query, 80),
                _preview(qp.lexical_query, 80),
            )

        yield StatusEvent(message=_prompts.STATUS_REVIEWING_DOCS, mode=self.name)

        embedding_service = EmbeddingService()
        reranker = RerankerClient() if search_settings.retriever_enable_reranker else None
        try:
            search_coros = [
                search_user_documents(
                    semantic_query=qp.semantic_query,
                    lexical_query=qp.lexical_query,
                    conversation_id=conversation_id,
                    limit=settings.max_user_doc_chunks,
                    alpha=settings.search_alpha,
                    embedding_service=embedding_service,
                    reranker=reranker,
                )
                for qp in query_set.queries
            ]
            search_tasks = []
            async with asyncio.TaskGroup() as task_group:
                for search_coro in search_coros:
                    search_tasks.append(task_group.create_task(search_coro))
            all_results = [task.result() for task in search_tasks]
        finally:
            if reranker is not None:
                await reranker.aclose()

        hits = _fuse_user_doc_rrf(all_results, max_chunks=settings.max_user_doc_chunks)

        if not hits:
            logger.info("mode=user_doc_retrieve: ingen treff funnet")
            for doc in ready_docs:
                doc.retrieved = True
            return

        doc_chunks: Dict[str, List] = defaultdict(list)
        for hit in hits:
            doc_chunks[hit.document_id].append(hit)

        yield StatusEvent(message=_prompts.STATUS_EVALUATING_DOCS, mode=self.name)

        for doc in ready_docs:
            doc.retrieved = True
            chunks_for_doc = doc_chunks.get(doc.id, [])

            if chunks_for_doc:
                doc.chunks = [
                    UserDocChunk(
                        chunk_index=hit.chunk_index,
                        text=hit.text[:settings.user_doc_chunk_max_chars],
                        score=hit.score,
                    )
                    for hit in chunks_for_doc
                ]
                doc.summary = await self._generate_summary(manager, chat_history, doc.chunks, doc.filename)
                logger.info(
                    "mode=user_doc_retrieve doc=%s chunks=%d summary_len=%d",
                    doc.filename, len(doc.chunks), len(doc.summary),
                )

        logger.info(
            "mode=user_doc_retrieve done hits=%s queries=%s",
            len(hits), len(query_set.queries),
        )

    async def _lazy_ingest(self, docs: List[UserDoc], conversation_id: str) -> None:
        """Ingest documents into Weaviate that haven't been ingested yet."""
        docs_by_id = {doc.id: doc for doc in docs}
        db_docs = [
            db_doc
            async for db_doc in UserDocument.objects.filter(
                id__in=[doc.id for doc in docs],
                status=UserDocument.Status.READY,
            ).only("id", "filename", "extracted_text", "weaviate_ingested").aiterator()
        ]
        if not db_docs:
            return

        embedding_service = EmbeddingService()
        vector_store = UserDocumentStore()

        for db_doc in db_docs:
            doc = docs_by_id[str(db_doc.id)]
            if db_doc.weaviate_ingested:
                doc.weaviate_ingested = True
                continue
            await ingest_user_document(
                conversation_id=conversation_id,
                document_id=doc.id,
                filename=db_doc.filename,
                text=db_doc.extracted_text,
                embedding_service=embedding_service,
                vector_store=vector_store,
            )
            db_doc.weaviate_ingested = True
            await db_doc.asave(update_fields=["weaviate_ingested"])
            doc.weaviate_ingested = True
            logger.info("mode=user_doc_retrieve lazy_ingest doc=%s", doc.filename)

    async def _generate_summary(
        self,
        manager: "ChatManager",
        chat_history: ChatHistory,
        chunks: List,
        filename: str,
    ) -> str:
        """Generate a summary of the document based on retrieved chunks."""
        prompt = _prompts.build_summary_input(filename, chunks)

        summary_model = MODE_MODELS["user_doc_summary"][chat_history.metadata.quality_mode]
        try:
            _, payload = await manager._call_structured_response(
                chat_history=chat_history,
                schema_model=SummaryPayload,
                schema_name="summary",
                instructions=_prompts.SUMMARY_INSTRUCTIONS,
                input_items=[{"role": "user", "content": prompt}],
                model=summary_model,
                store=False,
            )
            return payload.summary
        except StructuredOutputError as exc:
            logger.warning("mode=user_doc_retrieve summary fallback for %s: %s", filename, exc)
            return chunks[0].text[:settings.user_doc_chunk_max_chars] if chunks else ""

    async def _update_document_references(self, conversation_id: str, document_ids: List[str]) -> None:
        """Oppdater last_referenced_at og message_count for dokumenter som ble brukt."""
        conversation = await ChatConversation.objects.aget(pk=conversation_id)
        message_count = await conversation.messages.filter(channel="ui").acount()
        now = timezone.now()
        await UserDocument.objects.filter(id__in=document_ids).aupdate(
            last_referenced_at=now,
            message_count_at_reference=message_count,
        )
        logger.debug("Oppdaterte referanser for %d dokumenter", len(document_ids))
