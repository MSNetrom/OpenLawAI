"""Chunk-level Reciprocal Rank Fusion (RRF) with multiplicative link graph boosting.

Merges results from multiple search queries into a ranked, trimmed list of
LegalDocumentModel objects using rank-based fusion (not score comparison).

Pipeline:
  1. Per-query multiplicative link graph boost → rank assignment per document type
  2. Chunk-level RRF across queries
  3. Merged multiplicative link graph boost across full pool
     (optional context_work_ids from prior rounds act as extra link sources)
  4. Aggregate chunks into documents, trim chunks per doc, compute doc score
  5. Scoring-mode-dependent selection:
     t1_per_query — best doc per query (sum of top-2 reranker scores), up to max_total_docs
     t2_aggregate — best aggregate RRF doc score, up to max_total_docs
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Literal

from django.apps import apps

from legal_pipeline.search_models import ChunkModel, HydratedChunk, LegalDocumentModel

logger = logging.getLogger(__name__)

T1_TOP_K = 2


async def fuse_multi_query(
    query_results: list[list[HydratedChunk]],
    max_total_docs: int,
    max_chunks_per_doc: int,
    scoring_mode: Literal["t1_per_query", "t2_aggregate"] = "t2_aggregate",
    rrf_k: int = 60,
    link_factor: float = 0.3,
    query_types: list[str] | None = None,
    context_work_ids: list[int] | None = None,
) -> list[LegalDocumentModel]:
    """Fuse results from multiple search queries into ranked, trimmed documents.

    Args:
        max_total_docs: Maximum number of documents to return (flat budget, no per-type constraint).
        scoring_mode: Selection strategy after RRF scoring.
            t1_per_query: For each query, pick the best doc by sum of top-2 reranker scores.
            t2_aggregate: Pick the best docs by aggregate RRF doc score.
        query_types: Optional per-query type labels (aligned 1:1 with query_results).
        context_work_ids: Optional list of work_ids from a previous round.  When
            provided, these are included as potential link *sources* in Step 3
            (merged link boost) so that round-1 documents can boost round-2 docs.
    """
    rel_model = apps.get_model("legaldb", "DocumentRelationship")

    # Build chunk registry: best version of each unique chunk across all queries
    chunk_registry: dict[str, HydratedChunk] = {}
    for query_chunks in query_results:
        for c in query_chunks:
            existing = chunk_registry.get(c.vector_id)
            if existing is None or c.reranker_score > existing.reranker_score:
                chunk_registry[c.vector_id] = c

    if not chunk_registry:
        return []

    # --- Step 1: Per-query multiplicative link graph boost + rank assignment ---
    query_ranks: list[dict[str, int]] = []
    for query_chunks in query_results:
        if not query_chunks:
            query_ranks.append({})
            continue

        work_ids = list({c.work_id for c in query_chunks})
        work_ref_ids = {c.work_id: c.work_ref_id for c in query_chunks}
        link_norms, _ = await _compute_link_norms(
            work_ids, work_ref_ids, rel_model,
        )

        by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for c in query_chunks:
            factor = 1.0 + link_factor * link_norms.get(c.work_id, 0.0)
            by_type[c.document_type].append((c.vector_id, c.reranker_score * factor))

        ranks: dict[str, int] = {}
        for type_entries in by_type.values():
            type_entries.sort(key=lambda x: x[1], reverse=True)
            for rank, (vid, _) in enumerate(type_entries, 1):
                ranks[vid] = rank
        query_ranks.append(ranks)

    # --- Step 2: Chunk-level RRF ---
    rrf_scores: dict[str, float] = {}
    for vid in chunk_registry:
        score = sum(
            1.0 / (rrf_k + q_ranks[vid])
            for q_ranks in query_ranks
            if vid in q_ranks
        )
        rrf_scores[vid] = score

    # --- Step 3: Merged multiplicative link graph boost ---
    all_work_ids = list({c.work_id for c in chunk_registry.values()})
    all_work_ref_ids = {c.work_id: c.work_ref_id for c in chunk_registry.values()}
    merged_link_norms, merged_sources = await _compute_link_norms(
        all_work_ids, all_work_ref_ids, rel_model,
        context_work_ids=context_work_ids,
    )

    rrf_only: dict[str, float] = dict(rrf_scores)

    for vid in rrf_scores:
        work_id = chunk_registry[vid].work_id
        norm = merged_link_norms.get(work_id, 0.0)
        rrf_scores[vid] *= (1.0 + link_factor * norm)

    # --- Step 4: Aggregate chunks into documents ---
    doc_vid_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for vid, score in rrf_scores.items():
        doc_vid_scores[chunk_registry[vid].work_ref_id].append((vid, score))

    documents: list[LegalDocumentModel] = []
    for work_ref_id, vid_scores in doc_vid_scores.items():
        vid_scores.sort(key=lambda x: x[1], reverse=True)
        total_found = len(vid_scores)

        score_window = int(max_chunks_per_doc * 2)
        kept = vid_scores[:max_chunks_per_doc]
        scored = vid_scores[:score_window]
        doc_score = sum(s for _, s in scored)
        rrf_doc_score = sum(rrf_only[vid] for vid, _ in scored)

        first = chunk_registry[kept[0][0]]
        link_norm_val = merged_link_norms.get(first.work_id, 0.0)
        link_score_display = link_norm_val if link_norm_val > 0 else None
        sources = merged_sources.get(first.work_id, [])

        chunks = [
            ChunkModel(
                chunk_ref_id=chunk_registry[vid].chunk_ref_id,
                vector_id=vid,
                text=chunk_registry[vid].text,
                section_id=chunk_registry[vid].section_id,
                lovdata_url=chunk_registry[vid].lovdata_url,
                heading=chunk_registry[vid].heading,
                score=score,
                metadata=chunk_registry[vid].metadata,
            )
            for vid, score in kept
        ]

        documents.append(LegalDocumentModel(
            work_id=first.work_id,
            work_ref_id=work_ref_id,
            title=first.title,
            document_type=first.document_type,
            score=doc_score,
            rrf_score=rrf_doc_score,
            total_chunk_count=total_found,
            chunk_count=len(chunks),
            in_force=first.in_force,
            version_id=first.version_id,
            version_label=first.version_label,
            link_score=link_score_display,
            link_sources=sources,
            chunks=chunks,
        ))

    # --- Step 5: Selection by scoring mode (flat budget, no per-type constraint) ---
    doc_map = {d.work_ref_id: d for d in documents}

    if scoring_mode == "t1_per_query":
        selected = _select_t1_per_query(
            query_results, documents, doc_map, max_total_docs, query_types,
        )
    else:
        selected = _select_t2_aggregate(documents, max_total_docs)

    for ref_id in selected:
        doc_map[ref_id].selection_tier = 1 if scoring_mode == "t1_per_query" else 2

    trimmed = [doc_map[ref_id] for ref_id in selected]
    trimmed.sort(key=lambda d: d.score, reverse=True)
    logger.info(
        "fuse_multi_query mode=%s queries=%d chunks=%d docs_out=%d",
        scoring_mode, len(query_results), len(chunk_registry), len(trimmed),
    )
    return trimmed


def _select_t1_per_query(
    query_results: list[list[HydratedChunk]],
    documents: list[LegalDocumentModel],
    doc_map: dict[str, LegalDocumentModel],
    max_total_docs: int,
    query_types: list[str] | None,
) -> list[str]:
    """Select best doc per document type per query by sum of top-k reranker scores."""
    doc_types = {ref_id: doc.document_type for ref_id, doc in doc_map.items()}

    query_doc_scores: list[dict[str, float]] = []
    for qi, q_chunks in enumerate(query_results):
        per_doc: dict[str, list[float]] = defaultdict(list)
        for c in q_chunks:
            per_doc[c.work_ref_id].append(c.reranker_score)
        scores: dict[str, float] = {}
        for ref_id, chunk_scores in per_doc.items():
            chunk_scores.sort(reverse=True)
            scores[ref_id] = sum(chunk_scores[:T1_TOP_K])
        query_doc_scores.append(scores)

    selected: list[str] = []
    valid_refs = set(doc_map.keys())
    for qi, scores in enumerate(query_doc_scores):
        if len(selected) >= max_total_docs:
            break
        qt_label = query_types[qi] if query_types and qi < len(query_types) else "?"
        best_per_type: dict[str, tuple[str, float]] = {}
        for ref_id, s in scores.items():
            if ref_id not in valid_refs or ref_id in selected:
                continue
            dt = doc_types[ref_id]
            if dt not in best_per_type or s > best_per_type[dt][1]:
                best_per_type[dt] = (ref_id, s)
        for dt, (ref_id, s) in sorted(best_per_type.items(), key=lambda x: x[1][1], reverse=True):
            if len(selected) >= max_total_docs:
                break
            selected.append(ref_id)
            logger.info(
                "fusion t1 query=%d doc=%s (%s) query_type=%s type=%s top%d=%.4f",
                qi, ref_id, doc_map[ref_id].title[:50], qt_label, dt, T1_TOP_K, s,
            )
    return selected


def _select_t2_aggregate(
    documents: list[LegalDocumentModel],
    max_total_docs: int,
) -> list[str]:
    """Select top docs by aggregate RRF score (global, no per-type)."""
    docs_by_score = sorted(documents, key=lambda d: d.score, reverse=True)
    selected: list[str] = []
    for doc in docs_by_score:
        if len(selected) >= max_total_docs:
            break
        selected.append(doc.work_ref_id)
        logger.info(
            "fusion t2 doc=%s (%s) score=%.4f",
            doc.work_ref_id, doc.title[:50], doc.score,
        )
    return selected


async def _compute_link_norms(
    work_ids: list[int],
    work_ref_ids: dict[int, str],
    rel_model: Any,
    context_work_ids: list[int] | None = None,
) -> tuple[dict[int, float], dict[int, list[str]]]:
    """Compute normalized link factors (0.0-1.0) for documents referenced by work_ids.

    Uses unique source document count normalized by pool size (log-scaled).

    When context_work_ids is provided, those IDs are included as potential link
    *sources* (FROM side) but only work_ids are boosted (TO side).  This allows
    round-1 documents to boost round-2 documents via the link graph.

    Returns:
        norms: work_id → normalized link factor (0.0 to 1.0)
        sources: work_id → sorted list of source work_ref_ids that link to it
    """
    if not work_ids:
        return {}, {}

    from_ids = list(set(work_ids) | set(context_work_ids or []))
    all_size = len(set(from_ids))
    pool_norm = math.log1p(all_size)

    rel_rows = [
        row async for row in rel_model.objects.filter(
            from_work_id__in=from_ids,
            to_work_id__in=work_ids,
        ).values("from_work_id", "to_work_id")
    ]

    link_source_set: dict[int, set[int]] = {}
    for row in rel_rows:
        to_wid = row["to_work_id"]
        from_wid = row["from_work_id"]
        if to_wid not in link_source_set:
            link_source_set[to_wid] = set()
        link_source_set[to_wid].add(from_wid)

    norms: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for wid, source_wids in link_source_set.items():
        unique_count = len(source_wids)
        norm = math.log1p(unique_count) / pool_norm
        if norm > 1.0:
            norm = 1.0
        norms[wid] = norm
        sources[wid] = sorted(
            work_ref_ids[swid] for swid in source_wids if swid in work_ref_ids
        )

    return norms, sources
