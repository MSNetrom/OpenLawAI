"""RetrieveMode - Dual pipeline retrieval using OpenAI Responses API.

Runs two independent parallel pipelines:
  T1 (per-query winners): best doc per query by reranker score
  T2 (aggregate RRF):     best docs by aggregate RRF score across all queries

Each pipeline generates up to 7 queries, searches, fuses, and runs its own
refine loop (LLM drops to 7 docs, generates new queries, searches again).
Results are merged by work_ref_id with chunk dedup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Literal, Union

from config.model_routing import MODE_MODELS
from agents.mode_base import Mode
from agents.models import (
    ChatHistory,
    ErrorEvent,
    HydratedChunk,
    LegalDocumentModel,
    ModeName,
    ModeResult,
    QualityModeName,
    RetrievalQueryPayload,
    RetrievalQuerySet,
    RetrievalRefinePayload,
    StatusEvent,
    StreamEvent,
    settings,
)
from agents.shared import (
    StructuredOutputError,
    _preview,
    _trim_documents_for_context,
    _ui_user_turn_count,
)
from legal_pipeline.fusion import fuse_multi_query

if TYPE_CHECKING:
    from chat_manager import ChatManager

logger = logging.getLogger(__name__)

MAX_DOCS_PER_PIPELINE = 7


from agents.locale import load_prompts

_prompts = load_prompts("agents.retrieve.languages")


# --- Pipeline runner ---

async def _run_pipeline(
    manager: "ChatManager",
    chat_history: ChatHistory,
    pipeline_type: Literal["t1", "t2"],
    query_model: str,
    refine_model: str,
    max_refines: int,
    seed_docs: list[LegalDocumentModel] | None = None,
) -> tuple[list[LegalDocumentModel], str, int]:
    """Run a single retrieval pipeline (query gen → search → fuse → refine loop).

    Returns (docs, coverage_summary, rounds_run) with up to MAX_DOCS_PER_PIPELINE documents.
    """
    scoring_mode = "t1_per_query" if pipeline_type == "t1" else "t2_aggregate"
    tag = f"pipeline={pipeline_type}"
    seeded_ref_ids = {doc.work_ref_id for doc in seed_docs} if seed_docs is not None else set()

    # --- Initial search ---
    base_params: Dict[str, Any] = {"in_force": "true" if settings.retrieval_in_force_only else "false"}
    for doc_type in ("law", "forskrift"):
        base_params[f"{doc_type}_chunks"] = str(settings.search_chunks)
    initial_queries: list[RetrievalQueryPayload] = []
    if seed_docs is None:
        query_prompt = _prompts.build_query_prompt()
        query_context = _prompts.user_doc_context(chat_history) + await manager._context_for_mode(chat_history)

        try:
            _, query_set = await manager._call_structured_response(
                chat_history=chat_history,
                schema_model=RetrievalQuerySet,
                schema_name="retrieval_query_set",
                instructions=query_prompt,
                input_items=query_context,
                model=query_model,
                store=False,
            )
        except StructuredOutputError as exc:
            logger.warning("%s invalid query set: %s", tag, exc)
            return [], "", 0

        initial_queries = query_set.queries
        for i, qp in enumerate(initial_queries):
            logger.info(
                "%s query %d/%d [%s]: semantic=%s lexical=%s",
                tag, i + 1, len(initial_queries), qp.query_type,
                _preview(qp.semantic_query, 80),
                _preview(qp.lexical_query, 80),
            )

        query_chunks, query_type_labels = await _search_queries(
            manager, chat_history, initial_queries, base_params,
        )

        fused = await fuse_multi_query(
            query_chunks,
            max_total_docs=MAX_DOCS_PER_PIPELINE * 2,
            max_chunks_per_doc=settings.max_chunks_per_doc,
            scoring_mode=scoring_mode,
            query_types=query_type_labels,
        )
        pool: Dict[str, LegalDocumentModel] = {
            doc.work_ref_id: _sync_document_metadata(doc) for doc in fused
        }
        logger.info("%s initial search queries=%s pool=%s", tag, len(initial_queries), len(pool))
    else:
        pool = {
            doc.work_ref_id: _sanitize_seed_document(doc)
            for doc in seed_docs
        }
        logger.info("%s seeded pool=%s", tag, len(pool))

    dropped_ref_ids: set[str] = set()
    rounds_run = 0

    # --- Refine loop ---
    new_refine_queries: List[RetrievalQueryPayload] = []
    for round_index in range(max_refines):
        rounds_run += 1
        pool_trimmed = _trim_documents_for_context(
            [d.model_dump() for d in pool.values()],
            max_total_docs=MAX_DOCS_PER_PIPELINE * 2,
            max_chunks_per_doc=settings.max_chunks_per_doc,
        )

        if round_index == 0 and initial_queries:
            queries_summary = "\n".join(
                f"  {i}. [{qp.query_type}] Semantic: {qp.semantic_query}\n     Lexical: {qp.lexical_query}"
                for i, qp in enumerate(initial_queries, 1)
            )
        elif round_index == 0:
            queries_summary = _prompts.QUERIES_SUMMARY_PREVIOUS_ROUND
        else:
            queries_summary = "\n".join(
                f"  {i}. [{qp.query_type}] Semantic: {qp.semantic_query}\n     Lexical: {qp.lexical_query}"
                for i, qp in enumerate(new_refine_queries, 1)
            )

        pool_count = len(pool)
        occupancy_str = _prompts.occupancy_status(pool_count, MAX_DOCS_PER_PIPELINE)

        refine_prompt = _prompts.build_refine_prompt(
            max_docs=MAX_DOCS_PER_PIPELINE,
            round_index=round_index,
            max_refines=max_refines,
            occupancy_str=occupancy_str,
            queries_summary=queries_summary,
            pool_trimmed_json=json.dumps(pool_trimmed, ensure_ascii=False),
        )

        logger.info("%s refine start round=%s pool=%s", tag, round_index + 1, pool_count)

        refine_context = _prompts.user_doc_context(chat_history) + await manager._context_for_mode(chat_history)
        try:
            _, decision = await manager._call_structured_response(
                chat_history=chat_history,
                schema_model=RetrievalRefinePayload,
                schema_name="retrieval_refine",
                instructions=refine_prompt,
                input_items=refine_context,
                model=refine_model,
                store=False,
            )
        except StructuredOutputError as exc:
            logger.warning("%s refine invalid output round=%s: %s", tag, round_index + 1, exc)
            break

        # Process drops
        dropped_ids = [wid for wid in dict.fromkeys(decision.drop_work_ref_ids) if wid in pool]
        new_refine_queries = decision.new_queries

        logger.info("%s round=%s dropped=%s new_queries=%s", tag, round_index + 1, len(dropped_ids), len(new_refine_queries))
        for wid in dropped_ids:
            logger.info("%s round=%s DROP %s (%s)", tag, round_index + 1, wid, pool[wid].title)
            del pool[wid]
            dropped_ref_ids.add(wid)

        if not new_refine_queries:
            break

        # --- Search with new queries ---
        base_params_refine: Dict[str, Any] = {**base_params}
        if dropped_ref_ids:
            base_params_refine["exclude_ref_ids"] = ",".join(dropped_ref_ids)

        new_chunks, new_qt_labels = await _search_queries(
            manager, chat_history, new_refine_queries, base_params_refine,
        )

        pool_vids = {c.vector_id for doc in pool.values() for c in doc.chunks}
        deduped = [[c for c in batch if c.vector_id not in pool_vids] for batch in new_chunks]

        all_new = [c for batch in deduped for c in batch]
        if all_new:
            new_fused = await fuse_multi_query(
                deduped,
                max_total_docs=MAX_DOCS_PER_PIPELINE * 2,
                max_chunks_per_doc=settings.max_chunks_per_doc,
                scoring_mode=scoring_mode,
                query_types=new_qt_labels,
            )
            for doc in sorted(new_fused, key=lambda d: d.score, reverse=True):
                if doc.work_ref_id in pool:
                    _merge_document_evidence(pool[doc.work_ref_id], doc)
                elif len(pool) < MAX_DOCS_PER_PIPELINE * 2:
                    pool[doc.work_ref_id] = _sync_document_metadata(doc)

        logger.info("%s round=%s pool_after=%s", tag, round_index + 1, len(pool))

    # --- Final drop ---
    pool_trimmed_final = _trim_documents_for_context(
        [d.model_dump() for d in pool.values()],
        max_total_docs=MAX_DOCS_PER_PIPELINE * 2,
        max_chunks_per_doc=settings.max_chunks_per_doc,
    )

    final_prompt = _prompts.build_final_refine_prompt(
        max_docs=MAX_DOCS_PER_PIPELINE,
        pool_trimmed_json=json.dumps(pool_trimmed_final, ensure_ascii=False),
    )

    refine_context = _prompts.user_doc_context(chat_history) + await manager._context_for_mode(chat_history)
    try:
        _, final_decision = await manager._call_structured_response(
            chat_history=chat_history,
            schema_model=RetrievalRefinePayload,
            schema_name="retrieval_refine",
            instructions=final_prompt,
            input_items=refine_context,
            model=refine_model,
            store=False,
        )
    except StructuredOutputError as exc:
        logger.warning("%s final-drop invalid output: %s", tag, exc)
        final_decision = RetrievalRefinePayload(
            drop_work_ref_ids=[], new_queries=[], coverage_summary="",
        )

    for wid in dict.fromkeys(final_decision.drop_work_ref_ids):
        if wid in pool:
            logger.info("%s final-drop DROP %s (%s)", tag, wid, pool[wid].title)
            del pool[wid]

    result = sorted(
        (_sync_document_metadata(doc) for doc in pool.values()),
        key=lambda d: (d.work_ref_id in seeded_ref_ids, d.score),
        reverse=True,
    )[:MAX_DOCS_PER_PIPELINE]
    coverage = final_decision.coverage_summary
    logger.info("%s done docs=%s coverage=%s", tag, len(result), _preview(coverage, 80))
    return result, coverage, rounds_run


async def _search_queries(
    manager: "ChatManager",
    chat_history: ChatHistory,
    queries: List[RetrievalQueryPayload],
    base_params: Dict[str, Any],
) -> tuple[list[list[HydratedChunk]], list[str]]:
    """Run search for a list of queries in parallel. Returns (chunks_per_query, query_type_labels)."""
    search_coros = []
    for qp in queries:
        qp_params = {**base_params}
        if qp.query_type == "targeted":
            qp_params["alpha"] = "0.1"
        elif qp.query_type == "conceptual":
            qp_params["alpha"] = "0.8"
        search_coros.append(
            manager.legal_search.search(
                lexical_query=qp.lexical_query,
                semantic_query=qp.semantic_query,
                extra_params=qp_params,
            )
        )
    chat_history.metadata.tool_calls.search_documents += len(search_coros)

    search_tasks = []
    async with asyncio.TaskGroup() as tg:
        for coro in search_coros:
            search_tasks.append(tg.create_task(coro))

    query_chunks = [
        [HydratedChunk.model_validate(c) for c in task.result()["chunks"]]
        for task in search_tasks
    ]
    query_type_labels = [qp.query_type for qp in queries]
    return query_chunks, query_type_labels


def _merge_pipelines(
    t1_docs: list[LegalDocumentModel],
    t2_docs: list[LegalDocumentModel],
) -> list[LegalDocumentModel]:
    """Merge T1 and T2 results, deduplicating by work_ref_id with chunk merge.

    Docs found by both pipelines get all unique chunks from both (no cap).
    """
    merged: Dict[str, LegalDocumentModel] = {}

    for doc in t1_docs:
        doc.selection_tier = 1
        merged[doc.work_ref_id] = _sync_document_metadata(doc)

    for doc in t2_docs:
        if doc.work_ref_id in merged:
            existing = merged[doc.work_ref_id]
            existing.selection_tier = 3
            _merge_document_evidence(existing, doc)
        else:
            doc.selection_tier = 2
            merged[doc.work_ref_id] = _sync_document_metadata(doc)

    _tier_order = {3: 0, 1: 1, 2: 2}
    return sorted(
        (_sync_document_metadata(doc) for doc in merged.values()),
        key=lambda d: _tier_order[d.selection_tier],
    )


def _sync_document_metadata(doc: LegalDocumentModel) -> LegalDocumentModel:
    doc.chunks.sort(key=lambda c: c.score, reverse=True)
    doc.chunk_count = len(doc.chunks)
    doc.total_chunk_count = max(doc.total_chunk_count, doc.chunk_count)
    return doc


def _sanitize_seed_document(doc: LegalDocumentModel) -> LegalDocumentModel:
    seeded = LegalDocumentModel.model_validate(doc.model_dump())
    seeded.score = 0.0
    seeded.rrf_score = 0.0
    seeded.link_score = None
    seeded.link_sources = []
    seeded.selection_tier = None
    for chunk in seeded.chunks:
        chunk.score = 0.0
    seeded.total_chunk_count = len(seeded.chunks)
    return _sync_document_metadata(seeded)


def _merge_document_evidence(existing: LegalDocumentModel, incoming: LegalDocumentModel) -> LegalDocumentModel:
    existing_vids = {c.vector_id for c in existing.chunks}
    for chunk in incoming.chunks:
        if chunk.vector_id not in existing_vids:
            existing.chunks.append(chunk)
            existing_vids.add(chunk.vector_id)

    existing.score = _score_merged_chunks(existing.chunks)
    existing.rrf_score = max(existing.rrf_score, incoming.rrf_score)
    merged_link_scores = [score for score in (existing.link_score, incoming.link_score) if score is not None]
    existing.link_score = max(merged_link_scores) if merged_link_scores else None
    existing.link_sources = list(dict.fromkeys([*existing.link_sources, *incoming.link_sources]))
    existing.total_chunk_count = max(existing.total_chunk_count, incoming.total_chunk_count, len(existing.chunks))
    return _sync_document_metadata(existing)


def _score_merged_chunks(chunks) -> float:
    score_window = settings.max_chunks_per_doc * 2
    return sum(chunk.score for chunk in sorted(chunks, key=lambda c: c.score, reverse=True)[:score_window])


class RetrieveMode(Mode):
    """Dual-pipeline retrieval: T1 (per-query winners) + T2 (aggregate RRF) in parallel."""
    name: ModeName = "retrieve"
    models: Dict[QualityModeName, str] = MODE_MODELS["retrieve"]
    refine_models: Dict[QualityModeName, str] = MODE_MODELS["retrieve_refine"]
    max_refines_config: Dict[QualityModeName, int] = {"fast": 0, "thorough": 1}

    def get_max_refines(self, chat_history: ChatHistory) -> int:
        quality_mode: QualityModeName = chat_history.metadata.quality_mode
        return self.max_refines_config[quality_mode]

    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        chat_history.mode = self.name
        chat_history.metadata.mode_runs.retrieve += 1
        retrieve_model = self.get_model(chat_history)
        refine_model = self.refine_models[chat_history.metadata.quality_mode]
        max_refines = self.get_max_refines(chat_history)

        logger.info(
            "mode=retrieve start model=%s refine_model=%s max_refines=%s",
            retrieve_model, refine_model, max_refines,
        )

        chat_history.metadata.retrieval_calls += 1
        chat_history.metadata.retrieval_rounds = 0

        try:
            existing_legal_docs = chat_history.metadata.retrieval.results
            seeded = bool(existing_legal_docs)
            if seeded:
                max_refines += 1
                logger.info("mode=retrieve seeded from %d existing docs (max_refines=%d)", len(existing_legal_docs), max_refines)

            yield StatusEvent(message=_prompts.STATUS_SEARCHING, mode=self.name)

            seed_docs = [LegalDocumentModel.model_validate(d) for d in existing_legal_docs] if seeded else None
            async with asyncio.TaskGroup() as tg:
                t1_task = tg.create_task(_run_pipeline(
                    manager, chat_history, "t1", retrieve_model, refine_model, max_refines, seed_docs,
                ))
                t2_task = tg.create_task(_run_pipeline(
                    manager, chat_history, "t2", retrieve_model, refine_model, max_refines, seed_docs,
                ))
            t1_result, t1_coverage, t1_rounds = t1_task.result()
            t2_result, t2_coverage, t2_rounds = t2_task.result()

            merged = _merge_pipelines(t1_result, t2_result)
            logger.info(
                "mode=retrieve merged t1=%d t2=%d unique=%d",
                len(t1_result), len(t2_result), len(merged),
            )

            results = [d.model_dump() for d in merged]
            chat_history.metadata.retrieval.results = results
            chat_history.metadata.retrieval_rounds = t1_rounds + t2_rounds

            coverages = list(dict.fromkeys(c.strip() for c in (t1_coverage, t2_coverage) if c and c.strip()))
            chat_history.metadata.retrieval_coverage = " ".join(coverages) if coverages else None

            current_turn = _ui_user_turn_count(chat_history)
            for doc in results:
                chat_history.metadata.source_last_used.setdefault(doc["work_ref_id"], current_turn)

            logger.info("mode=retrieve done total=%s seeded=%s", len(results), seeded)
            yield ModeResult(next_mode="decide")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mode=retrieve failed, falling back to answer")
            yield ErrorEvent(detail=_prompts.ERROR_RETRIEVAL_FAILED)
            yield ModeResult(next_mode="answer")
