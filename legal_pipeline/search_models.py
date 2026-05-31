"""Shared data models for legal document search results.

These models are used by both the retrieval pipeline and the agent layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentType = Literal["law", "forskrift"]


class HydratedChunk(BaseModel):
    """Chunk with scores and parent document metadata, returned by the search API."""
    model_config = ConfigDict(extra="forbid")

    chunk_ref_id: int
    vector_id: str
    text: str
    section_id: str | None = None
    lovdata_url: str
    heading: str | None = None
    reranker_score: float
    hybrid_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    work_id: int
    work_ref_id: str
    title: str
    document_type: str
    in_force: bool
    version_id: int
    version_label: str | None = None


class ChunkModel(BaseModel):
    """A chunk of legal text from a document."""
    model_config = ConfigDict(extra="forbid")

    chunk_ref_id: int
    vector_id: str
    section_id: str | None = None
    lovdata_url: str
    heading: str | None = None
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalDocumentModel(BaseModel):
    """A legal document (law or forskrift) with its chunks."""
    model_config = ConfigDict(extra="forbid")

    work_id: int
    work_ref_id: str
    title: str
    document_type: DocumentType
    score: float
    rrf_score: float = 0.0
    total_chunk_count: int = 0
    chunk_count: int
    in_force: bool
    version_id: int | None = None
    version_label: str | None = None
    link_score: float | None = None
    link_sources: List[str] = Field(default_factory=list)
    selection_tier: int | None = None
    chunks: List[ChunkModel] = Field(default_factory=list)
