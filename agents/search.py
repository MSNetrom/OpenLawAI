"""Legal document search client."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict

import httpx

from agents.models import settings
from agents.shared import _preview

logger = logging.getLogger(__name__)


class LegalSearchClient:
    """Simple client for searching legal documents via the backend API."""
    
    def __init__(self, *, search_url: str, alpha: float, timeout_seconds: float) -> None:
        self.search_url = search_url
        self.alpha = alpha
        self.timeout_seconds = timeout_seconds
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
                self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
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

    async def search(
        self,
        *,
        lexical_query: str,
        semantic_query: str | None = None,
        extra_params: Dict[str, Any],
    ) -> Any:
        """Execute a search query against the legal documents API.

        Args:
            lexical_query: Query for BM25 keyword matching (required)
            semantic_query: Query for embedding/vector search (optional, uses lexical if not provided)
            extra_params: Additional search parameters
        """
        started = time.monotonic()
        params = {
            "q": lexical_query,
            "alpha": self.alpha,
            **extra_params,
        }
        # Add semantic query if provided (for hybrid search)
        if semantic_query:
            params["semantic_q"] = semantic_query

        max_retries = settings.search_max_retries
        for attempt in range(max_retries + 1):
            client = await self._get_client()
            request_error: Exception | None = None
            try:
                response = await client.get(self.search_url, params=params)
                response.raise_for_status()
                payload = response.json()
                elapsed_ms = int((time.monotonic() - started) * 1000)
                logger.info(
                    "legal_search ok status=%s ms=%s lexical=%s semantic=%s",
                    getattr(response, "status_code", "?"),
                    elapsed_ms,
                    _preview(lexical_query, 60),
                    _preview(semantic_query or "", 60),
                )
                return payload
            except httpx.RequestError as exc:
                await self._drop_client()
                request_error = exc
                retryable = True
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code in {408, 429} or exc.response.status_code >= 500
                if not retryable or attempt >= max_retries:
                    raise
                logger.warning(
                    "legal_search retrying status=%s attempt=%s/%s lexical=%s error=%s",
                    exc.response.status_code,
                    attempt + 1,
                    max_retries,
                    _preview(lexical_query, 60),
                    exc,
                )
                await asyncio.sleep(settings.search_retry_backoff_seconds * (attempt + 1))
                continue
            except json.JSONDecodeError as exc:
                await self._drop_client()
                retryable = True
                request_error = exc

            if attempt >= max_retries or not retryable:
                raise
            logger.warning(
                "legal_search retrying request attempt=%s/%s lexical=%s error=%s",
                attempt + 1,
                max_retries,
                _preview(lexical_query, 60),
                request_error,
            )
            await asyncio.sleep(settings.search_retry_backoff_seconds * (attempt + 1))

        raise RuntimeError("legal_search retry loop exhausted without returning")
