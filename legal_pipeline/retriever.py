from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from django.apps import apps

from legal_pipeline.search_models import ChunkModel, HydratedChunk, LegalDocumentModel
from config.app_settings import search_settings
from legal_pipeline.chunker import AsyncEmbeddingService, EmbeddingService
from legal_pipeline.reranker import RerankerClient
from legal_pipeline.weaviate_client import (
    ChunkHit,
    LegalChunkStore,
    UserDocumentChunkHit,
    UserDocumentStore,
)

logger = logging.getLogger(__name__)
MAX_RETRIEVAL_QUERY_CHARS = 2000


def normalize_retrieval_query(query: str) -> str:
    normalized = " ".join((query or "").split())
    return normalized[:MAX_RETRIEVAL_QUERY_CHARS].strip()


class DjangoChunkRepository:
    """Provides access to ChunkRef + related Django models."""

    def __init__(self) -> None:
        self.chunk_model = apps.get_model("legaldb", "ChunkRef")
        self.section_model = apps.get_model("legaldb", "DocumentSection")

    async def hydrate_hits(self, hits: Iterable[ChunkHit]) -> List[tuple]:
        """Attach Django ORM objects to the incoming chunk hits."""
        hits = list(hits)
        if not hits:
            return []
        vector_ids = [hit.vector_id for hit in hits]
        queryset = self.chunk_model.objects.select_related("version__work", "section").filter(vector_store_id__in=vector_ids)
        lookup = {chunk.vector_store_id: chunk async for chunk in queryset.aiterator()}
        hydrated = []
        missing_vector_ids = []
        for hit in hits:
            chunk = lookup.get(hit.vector_id)
            if not chunk:
                missing_vector_ids.append(hit.vector_id)
                continue
            hydrated.append((hit, chunk))
        if missing_vector_ids:
            logger.error(
                "Missing ChunkRef rows for %d Weaviate hits vector_ids=%s",
                len(missing_vector_ids),
                missing_vector_ids,
            )
        return hydrated


class DocumentScorer:
    """Scores documents based on chunk relevance + coverage."""

    def __init__(
        self,
        weight_chunk_score: float = 0.6,
        weight_chunk_count: float = 0.2,
        link_weight: float = 0.2,
        link_count_cap: int = 10,
        relationship_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weight_chunk_score = weight_chunk_score
        self.weight_chunk_count = weight_chunk_count
        self.link_weight = link_weight
        self.link_count_cap = link_count_cap
        self.relationship_weights = relationship_weights or {
            "based_on": 2.0,
            "changes": 1.0,
            "repeals": 1.0,
            "related": 1.0,
        }
        self.relationship_model = apps.get_model("legaldb", "DocumentRelationship")
        self.work_model = apps.get_model("legaldb", "DocumentWork")
        self.version_model = apps.get_model("legaldb", "DocumentVersion")

    async def _score_all(self, hydrated_hits: List[tuple]) -> List[LegalDocumentModel]:
        if not hydrated_hits:
            return []

        # Input: "hydrated" hits are (ChunkHit, ChunkRef) tuples where ChunkRef is a Django ORM object
        # linked to DocumentVersion -> DocumentWork. This lets us score at the *document* level instead
        # of scoring independent vector chunks.
        grouped: Dict[int, List[tuple]] = {}
        seed_ref_ids: Dict[int, str] = {}
        for hit, chunk in hydrated_hits:
            work_id = chunk.version.work_id
            if work_id not in grouped:
                grouped[work_id] = []
            grouped[work_id].append((hit, chunk))
            if work_id not in seed_ref_ids:
                seed_ref_ids[work_id] = chunk.version.work.ref_id

        result_map: Dict[int, LegalDocumentModel] = {}
        for work_id, entries in grouped.items():
            # Chunk relevance signal: average similarity / reranker score across matched chunks.
            chunk_scores = [hit.score for hit, _ in entries]
            avg_chunk_score = sum(chunk_scores) / len(chunk_scores)

            # Coverage signal: more matching chunks suggests higher relevance, but with diminishing returns.
            chunk_count_score = math.log1p(len(entries))
            chunk_count_score /= math.log1p(100)  # Normalize

            # Use any chunk to get document/version metadata (all chunks in `entries` belong to same work/version).
            chunk_obj = entries[0][1]
            work = chunk_obj.version.work
            version = chunk_obj.version

            # Final score is a weighted sum of: relevance (avg chunk) + coverage (chunk count).
            total = (
                self.weight_chunk_score * avg_chunk_score
                + self.weight_chunk_count * chunk_count_score
            )

            # Build the document-level payload, including the best matching chunks (sorted by chunk score).
            doc_chunks: List[ChunkModel] = []
            for hit, chunk in entries:
                section = chunk.section
                section_id = section.section_id if section else hit.section_id
                # ref_id contains the Lovdata URL path (e.g., "NL/lov/2005-06-17-62/§1-1")
                lovdata_url = section.ref_id if section else hit.ref_id
                heading = section.heading if section else None
                doc_chunks.append(
                    ChunkModel(
                        chunk_ref_id=chunk.id,
                        vector_id=hit.vector_id,
                        text=hit.text,
                        section_id=section_id,
                        lovdata_url=lovdata_url,
                        heading=heading,
                        score=hit.score,
                        metadata=hit.metadata,
                    )
                )
            doc_chunks.sort(key=lambda c: c.score, reverse=True)
            result_map[work_id] = LegalDocumentModel(
                work_id=work.id,
                work_ref_id=work.ref_id,
                title=work.title or work.short_title or "",
                document_type=work.document_type,
                score=total,
                chunk_count=len(entries),
                in_force=version.in_force,
                version_id=version.id,
                version_label=version.version_label,
                chunks=doc_chunks,
            )

        seed_work_ids = list(grouped.keys())
        rel_queryset = self.relationship_model.objects.filter(from_work_id__in=seed_work_ids).values(
            "from_work_id",
            "to_work_id",
            "relation_type",
        )
        rel_rows = [row async for row in rel_queryset]
        link_weight_sum: Dict[int, float] = {}
        link_sources: Dict[int, set[str]] = {}
        for row in rel_rows:
            relation_type = row["relation_type"]
            weight = self.relationship_weights[relation_type]
            to_work_id = row["to_work_id"]
            from_work_id = row["from_work_id"]
            if to_work_id not in link_weight_sum:
                link_weight_sum[to_work_id] = 0.0
            link_weight_sum[to_work_id] += weight
            if to_work_id not in link_sources:
                link_sources[to_work_id] = set()
            link_sources[to_work_id].add(seed_ref_ids[from_work_id])

        missing_work_ids = [work_id for work_id in link_weight_sum if work_id not in result_map]
        works = {work.id: work async for work in self.work_model.objects.filter(id__in=missing_work_ids)}
        versions: Dict[int, Any] = {}
        version_queryset = self.version_model.objects.filter(work_id__in=missing_work_ids).order_by(
            "work_id",
            "-created_at",
            "-id",
        )
        async for version in version_queryset:
            if version.work_id not in versions:
                versions[version.work_id] = version
        link_norm_divisor = math.log1p(self.link_count_cap)
        for work_id, link_score_raw in link_weight_sum.items():
            link_score_norm = math.log1p(link_score_raw) / link_norm_divisor
            if link_score_norm > 1.0:
                link_score_norm = 1.0
            sources = sorted(link_sources[work_id])
            if work_id in result_map:
                result = result_map[work_id]
                result.score += self.link_weight * link_score_norm
                result.link_score = link_score_norm
                result.link_sources = sources
                continue
            if work_id not in works or work_id not in versions:
                continue
            work = works[work_id]
            version = versions[work_id]
            result_map[work_id] = LegalDocumentModel(
                work_id=work.id,
                work_ref_id=work.ref_id,
                title=work.title or work.short_title or "",
                document_type=work.document_type,
                score=self.link_weight * link_score_norm,
                chunk_count=0,
                in_force=version.in_force,
                version_id=version.id,
                version_label=version.version_label,
                chunks=[],
                link_score=link_score_norm,
                link_sources=sources,
            )

        return list(result_map.values())

    async def score(self, hydrated_hits: List[tuple], top_k: int) -> List[LegalDocumentModel]:
        results = await self._score_all(hydrated_hits)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]



class DocumentRetriever:
    """High-level entrypoint combining Weaviate + Django metadata + scoring."""

    def __init__(
        self,
        weaviate_client: Optional[LegalChunkStore] = None,
        chunk_repository: Optional[DjangoChunkRepository] = None,
        embedding_service: Optional[AsyncEmbeddingService] = None,
        scorer: Optional[DocumentScorer] = None,
        reranker: Optional[RerankerClient] = None,
        enable_reranker: Optional[bool] = None,
    ) -> None:
        self.weaviate_client = weaviate_client or LegalChunkStore()
        self.chunk_repository = chunk_repository or DjangoChunkRepository()
        self.embedding_service = embedding_service or EmbeddingService()
        self.scorer = scorer or DocumentScorer()
        if enable_reranker is None:
            enable_reranker = search_settings.retriever_enable_reranker
        if reranker is not None:
            self.reranker = reranker
        elif enable_reranker:
            self.reranker = RerankerClient()
        else:
            self.reranker = None

    async def _maybe_rerank(self, query: str, hits: List[ChunkHit]) -> List[ChunkHit]:
        if not self.reranker:
            return hits
        if asyncio.iscoroutinefunction(getattr(self.reranker, "rerank", None)):
            return await self.reranker.rerank(query, hits)
        return self.reranker.rerank(query, hits)

    async def aretrieve_by_type(
        self,
        lexical_query: str,
        semantic_query: str,
        *,
        per_type: Dict[str, int],
        alpha: float | None = None,
        in_force_only: bool = False,
        exclude_ref_ids: Optional[Sequence[str]] = None,
    ) -> List[HydratedChunk]:
        """Retrieve flat hydrated chunks per document type for fusion.

        The server does mechanical retrieval only (Weaviate + rerank + hydrate).
        All scoring intelligence lives in the fusion module on the client side.

        Args:
            lexical_query: Query for BM25 keyword matching
            semantic_query: Query for embedding/vector search
            per_type: Chunk limits per document type, e.g. {"law": 64, "forskrift": 64}
            alpha: Hybrid search alpha (0=BM25, 1=vector)
            in_force_only: Only return documents currently in force
            exclude_ref_ids: Skip chunks from these work ref_ids (for refine loops)
        """
        lexical_query = normalize_retrieval_query(lexical_query)
        semantic_query = normalize_retrieval_query(semantic_query)
        if not lexical_query or not semantic_query:
            raise ValueError("Queries must contain searchable text.")
        alpha = alpha if alpha is not None else search_settings.search_alpha
        vector = await self.embedding_service.embed_query(semantic_query)
        all_chunks: List[HydratedChunk] = []
        for document_type, chunk_limit in per_type.items():
            hits = await self.weaviate_client.hybrid_search(
                query=lexical_query,
                vector=vector,
                limit=chunk_limit,
                alpha=alpha,
                document_types=[document_type],
                in_force_only=in_force_only,
                exclude_ref_ids=exclude_ref_ids,
            )
            hybrid_scores = {hit.vector_id: hit.score for hit in hits}
            hits = await self._maybe_rerank(semantic_query, hits)
            hydrated = await self.chunk_repository.hydrate_hits(hits)
            for hit, chunk in hydrated:
                section = chunk.section
                section_id = section.section_id if section else hit.section_id
                lovdata_url = section.ref_id if section else hit.ref_id
                heading = section.heading if section else None
                work = chunk.version.work
                version = chunk.version
                all_chunks.append(HydratedChunk(
                    chunk_ref_id=chunk.id,
                    vector_id=hit.vector_id,
                    text=hit.text,
                    section_id=section_id,
                    lovdata_url=lovdata_url,
                    heading=heading,
                    reranker_score=hit.score,
                    hybrid_score=hybrid_scores[hit.vector_id],
                    metadata=hit.metadata,
                    work_id=work.id,
                    work_ref_id=work.ref_id,
                    title=work.title or work.short_title or "",
                    document_type=work.document_type,
                    in_force=version.in_force,
                    version_id=version.id,
                    version_label=version.version_label,
                ))
        all_chunks.sort(key=lambda c: c.reranker_score, reverse=True)
        return all_chunks

    async def aretrieve(
        self,
        lexical_query: str,
        semantic_query: str | None = None,
        *,
        chunk_limit: int | None = None,
        document_limit: int | None = None,
        alpha: float | None = None,
        document_types: Optional[Sequence[str]] = None,
        in_force_only: bool = False,
    ) -> List[LegalDocumentModel]:
        """Retrieve documents using hybrid search.

        Args:
            lexical_query: Query for BM25 keyword matching
            semantic_query: Query for embedding/vector search (defaults to lexical_query)
        """
        semantic_query = normalize_retrieval_query(semantic_query or lexical_query)
        lexical_query = normalize_retrieval_query(lexical_query)
        if not lexical_query or not semantic_query:
            raise ValueError("Queries must contain searchable text.")
        chunk_limit = chunk_limit or search_settings.search_chunks
        document_limit = document_limit or search_settings.search_documents
        alpha = alpha if alpha is not None else search_settings.search_alpha
        vector = await self.embedding_service.embed_query(semantic_query)
        hits = await self.weaviate_client.hybrid_search(
            query=lexical_query,
            vector=vector,
            limit=chunk_limit,
            alpha=alpha,
            document_types=document_types,
            in_force_only=in_force_only,
        )
        hits = await self._maybe_rerank(semantic_query, hits)
        hydrated = await self.chunk_repository.hydrate_hits(hits)
        return await self.scorer.score(hydrated, top_k=document_limit)

    def retrieve(
        self,
        lexical_query: str,
        semantic_query: str | None = None,
        *,
        chunk_limit: int | None = None,
        document_limit: int | None = None,
        alpha: float | None = None,
        document_types: Optional[Sequence[str]] = None,
        in_force_only: bool = False,
    ) -> List[LegalDocumentModel]:
        """Sync wrapper for CLI usage."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("DocumentRetriever.retrieve() cannot be called from an async context; use aretrieve() instead.")
        return asyncio.run(
            self.aretrieve(
                lexical_query=lexical_query,
                semantic_query=semantic_query,
                chunk_limit=chunk_limit,
                document_limit=document_limit,
                alpha=alpha,
                document_types=document_types,
                in_force_only=in_force_only,
            )
        )


async def _rerank_user_doc_hits(
    reranker: RerankerClient,
    query: str,
    hits: List[UserDocumentChunkHit],
) -> List[UserDocumentChunkHit]:
    """Rerank user document hits using the reranker service."""
    texts = [h.text for h in hits]
    results = await reranker.rerank_texts(query, texts)
    scores_by_index = {r.index: r.relevance_score for r in results}
    for i, hit in enumerate(hits):
        hit.score = float(scores_by_index[i])
    return hits


async def search_user_documents(
    semantic_query: str,
    lexical_query: str,
    conversation_id: str,
    embedding_service: AsyncEmbeddingService,
    limit: int = 10,
    alpha: float = 0.5,
    search_client: Optional[UserDocumentStore] = None,
    reranker: Optional[RerankerClient] = None,
) -> List[UserDocumentChunkHit]:
    """
    Search user-uploaded documents for a specific conversation.

    Args:
        semantic_query: Natural language query for vector search
        lexical_query: Keywords for BM25 search
        conversation_id: UUID of the conversation (isolation key)
        embedding_service: Embedding service for vector search
        limit: Max chunks to return
        alpha: Hybrid search alpha (0=keyword, 1=vector)
        search_client: Optional pre-configured search client
        reranker: Optional reranker client for cross-encoder reranking

    Returns:
        List of matching chunks sorted by relevance (reranker score if available)
    """
    semantic_query = normalize_retrieval_query(semantic_query)
    lexical_query = normalize_retrieval_query(lexical_query)
    if not semantic_query or not lexical_query:
        return []
    client = search_client or UserDocumentStore()
    vector = await embedding_service.embed_query(semantic_query)

    hits = await client.hybrid_search(
        query=lexical_query,
        conversation_id=conversation_id,
        vector=vector,
        limit=limit,
        alpha=alpha,
    )
    user_document_model = apps.get_model("chatdb", "UserDocument")
    ready_document_ids = {
        str(doc.id)
        async for doc in user_document_model.objects.filter(
            conversation_id=conversation_id,
            status=user_document_model.Status.READY,
        ).only("id").aiterator()
    }
    hits = [hit for hit in hits if hit.document_id in ready_document_ids]
    if reranker and hits:
        hits = await _rerank_user_doc_hits(reranker, semantic_query, hits)
    hits.sort(key=lambda h: h.score, reverse=True)
    logger.info(
        "search_user_documents conversation=%s semantic=%s lexical=%s hits=%s reranked=%s",
        conversation_id, semantic_query[:40], lexical_query[:40], len(hits), reranker is not None,
    )
    return hits
