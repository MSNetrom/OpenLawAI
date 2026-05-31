from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(slots=True)
class DocumentMetadata:
    """Normalized metadata for a legal document."""

    ref_id: str
    dok_id: str
    legacy_id: Optional[str]
    title: Optional[str]
    short_title: Optional[str]
    document_type: str
    legal_source: str
    ministries: List[str] = field(default_factory=list)
    subunits: List[str] = field(default_factory=list)
    legal_areas: List[str] = field(default_factory=list)
    applies_to: List[str] = field(default_factory=list)
    authority_refs: List[str] = field(default_factory=list)
    date_in_force: List[str] = field(default_factory=list)
    date_of_publication: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    last_change_in_force: Optional[datetime] = None
    last_changed_by: Optional[str] = None
    misc_information: Optional[str] = None


@dataclass(slots=True)
class DocumentSection:
    """Represents a navigable section/paragraph extracted from the HTML."""

    section_id: str
    ref_id: str
    heading: Optional[str]
    text: str
    html: str
    level: int
    order: int
    parent_section_id: Optional[str] = None


@dataclass(slots=True)
class DocumentRelationship:
    """Captures pointers such as 'based on' or 'changes to' relationships."""

    relation_type: str
    target_ref_id: str
    description: Optional[str] = None


@dataclass(slots=True)
class ExtractedDocument:
    """Structured payload returned by the extractor module."""

    metadata: DocumentMetadata
    sections: List[DocumentSection]
    relationships: List[DocumentRelationship] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    """Chunk of text that will be embedded and indexed in Weaviate."""

    chunk_id: str
    section_id: str
    text: str
    order: int
    metadata: Dict[str, str]
    embedding: Optional[List[float]] = None
    vector_id: Optional[str] = None
