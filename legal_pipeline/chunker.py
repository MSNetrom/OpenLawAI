"""Text chunking and embedding utilities.

This module provides:
- LangChainSectionChunker: Uses LangChain text splitters for robust chunking
- EmbeddingService: Uses LangChain OpenAIEmbeddings for backend-agnostic embeddings
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Protocol, Sequence, runtime_checkable

if __package__ is None or __package__ == "":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import httpx

from config.app_settings import embedding_settings, search_settings
from legal_pipeline.data_structures import Chunk, DocumentSection, ExtractedDocument

try:  # pragma: no cover - optional import details depend on installed sdk version
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
except Exception:  # pragma: no cover
    APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = ()

logger = logging.getLogger(__name__)

# Export chunking constants for use by other modules
CHUNK_MAX_CHARS = search_settings.chunk_max_chars
CHUNK_OVERLAP = search_settings.chunk_overlap


@runtime_checkable
class AsyncEmbeddingService(Protocol):
    """Protocol for async embedding services."""

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for document texts (no instruction prefix)."""
        ...

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (with instruction prefix for retrieval)."""
        ...

# Separators optimized for Norwegian legal text
LEGAL_TEXT_SEPARATORS = [
    "\n\n\n",  # Major section breaks
    "\n\n",    # Paragraph breaks
    "\n",      # Line breaks
    "§",       # Section markers (keep with following text)
    ". ",      # Sentence boundaries
    ", ",      # Clause boundaries
    " ",       # Word boundaries
    "",        # Character fallback
]


class LangChainSectionChunker:
    """Splits document sections using LangChain's RecursiveCharacterTextSplitter.

    This provides robust handling of edge cases (empty text, Unicode, etc.)
    and respects natural text boundaries for Norwegian legal text.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ) -> None:
        chunk_size = search_settings.chunk_max_chars if chunk_size is None else chunk_size
        chunk_overlap = search_settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or LEGAL_TEXT_SEPARATORS
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def build_chunks(self, document: ExtractedDocument) -> List[Chunk]:
        """Build chunks from all sections of a document."""
        chunks: List[Chunk] = []
        for section in document.sections:
            section_chunks = self._chunk_section(section, document)
            chunks.extend(section_chunks)
        return chunks

    def _chunk_section(self, section: DocumentSection, document: ExtractedDocument) -> List[Chunk]:
        """Split a single section into chunks."""
        text = section.text.strip()
        if not text:
            return []

        # Use LangChain splitter
        text_chunks = self._splitter.split_text(text)

        # Build metadata for all chunks in this section
        metadata = {
            "ref_id": document.metadata.ref_id,
            "dok_id": document.metadata.dok_id,
            "section_id": section.section_id,
            "document_type": document.metadata.document_type,
            "legal_source": document.metadata.legal_source,
            "title": document.metadata.title or "",
            "short_title": document.metadata.short_title or "",
            "ministries": "|".join(document.metadata.ministries),
            "legal_areas": "|".join(document.metadata.legal_areas),
            "applies_to": "|".join(document.metadata.applies_to),
            "authority_refs": "|".join(document.metadata.authority_refs),
            "date_in_force": "|".join(document.metadata.date_in_force),
            "date_of_publication": (
                document.metadata.date_of_publication.isoformat()
                if document.metadata.date_of_publication
                else ""
            ),
            "misc_information": document.metadata.misc_information or "",
        }

        # Create Chunk objects
        chunk_list: List[Chunk] = []
        for order, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            # Include ref_id to ensure globally unique chunk_id across documents
            chunk_id = f"{document.metadata.ref_id}::{section.section_id}::chunk-{order}"
            chunk_list.append(
                Chunk(
                    chunk_id=chunk_id,
                    section_id=section.section_id,
                    text=chunk_text,
                    order=order,
                    metadata=metadata,
                )
            )

        return chunk_list


class EmbeddingService:
    """Embedding service using LangChain's OpenAIEmbeddings.

    Works with any OpenAI-compatible embedding endpoint:
    - vLLM (--task embed)
    - OpenAI API
    - Azure OpenAI
    - Any compatible endpoint

    Singleton: reuses one HTTP client to avoid resource leaks.

    Env vars:
        EMBEDDING_API_URL: Base URL (e.g. http://localhost:8001/v1)
        EMBEDDING_MODEL: Model name (e.g. BAAI/bge-m3)
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        resolved_base_url = (base_url or embedding_settings.embedding_api_url).rstrip("/")
        resolved_model = model or embedding_settings.embedding_model
        if hasattr(self, '_initialized'):
            if (self.base_url, self.model, self.timeout) != (resolved_base_url, resolved_model, timeout):
                raise RuntimeError(
                    "EmbeddingService singleton already initialized with different configuration"
                )
            return
        self._initialized = True

        self.base_url = resolved_base_url
        self.model = resolved_model
        self.timeout = timeout
        self.max_retries = embedding_settings.embedding_max_retries
        self.retry_backoff_seconds = embedding_settings.embedding_retry_backoff_seconds

        # LangChain OpenAIEmbeddings handles the API calls
        self._embeddings = OpenAIEmbeddings(
            base_url=self.base_url,
            model=self.model,
            api_key="not-needed",  # vLLM and local endpoints ignore this
            timeout=timeout,
        )

    @staticmethod
    def _is_retryable_embedding_error(exc: Exception) -> bool:
        retryable_types = tuple(
            exc_type for exc_type in (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
            if exc_type
        )
        if isinstance(exc, (httpx.RequestError, TimeoutError, OSError)):
            return True
        return isinstance(exc, retryable_types) if retryable_types else False

    async def _retry_async(self, operation, *, op_name: str):
        max_attempts = max(1, self.max_retries + 1)
        for attempt in range(max_attempts):
            try:
                return await operation()
            except Exception as exc:
                if attempt >= max_attempts - 1 or not self._is_retryable_embedding_error(exc):
                    raise
                delay = self.retry_backoff_seconds * (attempt + 1)
                logger.warning(
                    "Retrying embedding %s attempt=%s/%s delay=%.2fs error=%s",
                    op_name,
                    attempt + 1,
                    max_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    def _retry_sync(self, operation, *, op_name: str):
        max_attempts = max(1, self.max_retries + 1)
        for attempt in range(max_attempts):
            try:
                return operation()
            except Exception as exc:
                if attempt >= max_attempts - 1 or not self._is_retryable_embedding_error(exc):
                    raise
                delay = self.retry_backoff_seconds * (attempt + 1)
                logger.warning(
                    "Retrying embedding %s attempt=%s/%s delay=%.2fs error=%s",
                    op_name,
                    attempt + 1,
                    max_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embeddings for document texts (no instruction prefix)."""
        if not texts:
            return []
        return await self._retry_async(
            lambda: self._embeddings.aembed_documents(list(texts)),
            op_name="documents",
        )

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query with instruction prefix for retrieval."""
        formatted = f"Instruct: Given a legal query, retrieve relevant legal documents and passages\nQuery: {query}"
        result = await self._retry_async(
            lambda: self._embeddings.aembed_documents([formatted]),
            op_name="query",
        )
        if not result:
            raise RuntimeError("Embedding query returned no vectors")
        return result[0]

    def embed_sync(self, texts: Sequence[str]) -> List[List[float]]:
        """Synchronous embedding for batch processing."""
        if not texts:
            return []
        return self._retry_sync(
            lambda: self._embeddings.embed_documents(list(texts)),
            op_name="documents_sync",
        )
