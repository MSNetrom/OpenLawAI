"""Generic reranker client for OpenAI-compatible rerank endpoints.

Works with vLLM, Cohere, or any service implementing the /v1/rerank endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Sequence

import httpx

if TYPE_CHECKING:
    from legal_pipeline.retriever import ChunkHit

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking a single document."""
    index: int
    relevance_score: float


@dataclass
class RerankerClient:
    """Generic reranker using Cohere-compatible /v1/rerank endpoint.

    Works with:
    - vLLM (--task score)
    - Cohere API
    - Any OpenAI-compatible rerank endpoint

    Env vars:
        RERANK_API_URL: Base URL (e.g. http://localhost:8002/v1)
        RERANK_MODEL: Model name (e.g. Qwen/Qwen3-Reranker-4B)
    """

    base_url: str | None = None
    model: str | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        # Uses separate vLLM endpoint for reranking on port 8002
        self.base_url = (
            self.base_url
            or os.environ.get("RERANK_API_URL", "http://localhost:8002/v1")
        ).rstrip("/")
        self.model = self.model or os.environ.get("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None
        self._runtime_loop = None
        self._state_lock = threading.Lock()

    async def _ensure_async_state(self) -> None:
        loop = asyncio.get_running_loop()
        stale_client = None
        with self._state_lock:
            if self._runtime_loop is loop:
                return
            stale_client = self._client
            self._runtime_loop = loop
            self._client = None
            self._client_lock = asyncio.Lock()
        if stale_client is not None and not stale_client.is_closed:
            await stale_client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        await self._ensure_async_state()
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=self.timeout)
            return self._client

    async def _drop_client(self) -> None:
        await self._ensure_async_state()
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()

    async def aclose(self) -> None:
        await self._drop_client()

    async def rerank_texts(
        self, query: str, documents: Sequence[str]
    ) -> List[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.

        Returns:
            List of RerankResult with index and score.
        """
        if not documents:
            return []

        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": list(documents),
                },
            )
        except httpx.RequestError:
            await self._drop_client()
            raise
        response.raise_for_status()
        data = response.json()

        # Standard response: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        results = [
            RerankResult(index=r["index"], relevance_score=r["relevance_score"])
            for r in data["results"]
        ]

        logger.debug("rerank: query=%s docs=%d", query[:50], len(documents))
        return results

    async def rerank(self, query: str, hits: Sequence[ChunkHit]) -> List[ChunkHit]:
        """Rerank ChunkHit objects by relevance to query.

        Args:
            query: The search query.
            hits: Sequence of ChunkHit objects to rerank.

        Returns:
            List of ChunkHit objects with updated scores.
        """
        if not hits:
            return list(hits)

        documents = [hit.text for hit in hits]
        results = await self.rerank_texts(query, documents)

        scores_by_index = {r.index: r.relevance_score for r in results}

        for i, hit in enumerate(hits):
            score = scores_by_index[i]
            hit.metadata["reranker_model"] = self.model
            hit.metadata["reranker_score"] = float(score)
            hit.score = float(score)

        logger.debug("rerank hits: query=%s hits=%d", query[:50], len(hits))
        return list(hits)
