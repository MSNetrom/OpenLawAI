from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

if __package__ is None or __package__ == "":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import tiktoken
from django.db import transaction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from legal_pipeline.chunker import LangChainSectionChunker, EmbeddingService, CHUNK_MAX_CHARS, CHUNK_OVERLAP
from legal_pipeline.data_structures import Chunk, DocumentRelationship, ExtractedDocument
from legal_pipeline.extractors.lovdata import LovdataExtractor
from legal_pipeline.weaviate_client import LegalChunkStore, UserDocumentStore

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SavedDocument:
    version: Any
    sections_by_id: Dict[str, Any]


class IngestionBatchError(RuntimeError):
    """Raised after processing all files when one or more ingestions failed."""


class MetadataRepository(Protocol):
    """Interface hiding the concrete Django/ORM implementation."""

    def save_document(self, document: ExtractedDocument) -> SavedDocument: ...

    def save_chunks(self, saved: SavedDocument, chunks: List[Chunk], vector_ids: List[str]) -> None: ...


class VectorStore(Protocol):
    """Interface for the Weaviate client wrapper."""

    def upsert_chunks(self, chunks: List[Chunk]) -> List[str]: ...
    def delete_by_vector_ids(self, vector_ids: Sequence[str]) -> None: ...


class IngestionPipeline:
    """Coordinates extraction, chunking, embedding, and persistence."""

    def __init__(
        self,
        extractor: LovdataExtractor,
        chunker: LangChainSectionChunker,
        embedding_service: EmbeddingService,
        metadata_repository: MetadataRepository,
        vector_store: VectorStore,
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.metadata_repository = metadata_repository
        self.vector_store = vector_store

    def ingest_paths(self, paths: Iterable[Path]) -> None:
        failures: list[tuple[Path, str]] = []
        for path in paths:
            if path.is_dir():
                files = sorted(path.rglob("*.xml"))
            else:
                files = [path]

            for file_path in files:
                logger.info("Processing %s", file_path)
                try:
                    logger.debug("Parsing %s", file_path)
                    document = self.extractor.parse_file(file_path)
                    logger.info(
                        "Parsed %s (ref_id=%s, sections=%d)",
                        file_path,
                        document.metadata.ref_id,
                        len(document.sections),
                    )
                    saved = self.metadata_repository.save_document(document)
                    logger.debug("Saved metadata for %s", document.metadata.ref_id)
                    
                    # Chunk and embed document
                    chunks = self.chunker.build_chunks(document)
                    logger.info("Chunked %s into %d chunk(s)", document.metadata.ref_id, len(chunks))
                    if not chunks:
                        continue
                    
                    # Generate embeddings
                    texts = [chunk.text for chunk in chunks]
                    embeddings = self.embedding_service.embed_sync(texts)
                    for chunk, embedding in zip(chunks, embeddings):
                        chunk.embedding = embedding
                    
                    vector_ids = self.vector_store.upsert_chunks(chunks)
                    logger.info("Indexed %d chunk(s) for %s in Weaviate", len(vector_ids), document.metadata.ref_id)
                    try:
                        self.metadata_repository.save_chunks(saved, chunks, vector_ids)
                    except Exception:
                        try:
                            self.vector_store.delete_by_vector_ids(vector_ids)
                        except Exception:
                            logger.exception(
                                "Rollback failed while deleting %d vector(s) for %s after metadata save error",
                                len(vector_ids),
                                document.metadata.ref_id,
                            )
                        raise
                except Exception:
                    logger.exception("Failed to ingest %s", file_path)
                    failures.append((file_path, "failed"))
                    continue
        if failures:
            failed_paths = ", ".join(str(path) for path, _reason in failures[:10])
            raise IngestionBatchError(
                f"Ingestion completed with {len(failures)} failed file(s): {failed_paths}"
            )


class DjangoMetadataRepository:
    """Repository that persists legal document metadata to Django ORM (LegalDocument model)."""

    work_type_map = {
        "law": "law",
        "forskrift": "forskrift",
        "other": "other",
    }

    relation_map = {
        "BASED_ON": "based_on",
        "CHANGES": "changes",
        "REPEALS": "repeals",
        "RELATED": "related",
    }

    def __init__(self) -> None:
        self._models = None

    def _get_models(self):
        if self._models:
            return self._models

        from django.apps import apps

        models = {
            "LegalSource": apps.get_model("legaldb", "LegalSource"),
            "LegalArea": apps.get_model("legaldb", "LegalArea"),
            "Organization": apps.get_model("legaldb", "Organization"),
            "DocumentWork": apps.get_model("legaldb", "DocumentWork"),
            "DocumentOrganizationRole": apps.get_model("legaldb", "DocumentOrganizationRole"),
            "DocumentVersion": apps.get_model("legaldb", "DocumentVersion"),
            "DocumentSection": apps.get_model("legaldb", "DocumentSection"),
            "DocumentRelationship": apps.get_model("legaldb", "DocumentRelationship"),
            "ChunkRef": apps.get_model("legaldb", "ChunkRef"),
        }

        self._models = models
        return models

    def save_document(self, document: ExtractedDocument) -> SavedDocument:
        models = self._get_models()
        metadata = document.metadata

        with transaction.atomic():
            legal_source = self._get_legal_source(models["LegalSource"], metadata.legal_source)
            work = self._upsert_work(models, document, legal_source)
            version = self._upsert_version(models["DocumentVersion"], work, metadata)
            sections = self._sync_sections(models["DocumentSection"], version, document)
            self._sync_relationships(models, work, document.relationships)
        return SavedDocument(version=version, sections_by_id=sections)

    def _get_legal_source(self, LegalSource, code: Optional[str]):
        if not code:
            return None
        obj, _ = LegalSource.objects.get_or_create(code=code, defaults={"name": code})
        return obj

    def _upsert_work(self, models, document: ExtractedDocument, legal_source):
        metadata = document.metadata
        DocumentWork = models["DocumentWork"]
        defaults = {
            "legacy_id": metadata.legacy_id or "",
            "legal_source": legal_source,
            "document_type": self._normalize_document_type(DocumentWork, metadata.document_type),
            "title": metadata.title or "",
            "short_title": metadata.short_title or "",
            "language": "",
            "applies_to": metadata.applies_to,
            "metadata": {"raw": self._json_safe(asdict(metadata))},
            "date_in_force": self._first_date(metadata.date_in_force),
            "date_of_publication": self._ensure_aware(metadata.date_of_publication),
            "misc_information": metadata.misc_information or "",
        }
        work, _ = DocumentWork.objects.update_or_create(ref_id=metadata.ref_id, defaults=defaults)

        self._assign_legal_areas(models["LegalArea"], work, metadata.legal_areas)
        self._assign_organizations(
            models["Organization"], models["DocumentOrganizationRole"], work, metadata.ministries, "ministry"
        )
        self._assign_organizations(
            models["Organization"], models["DocumentOrganizationRole"], work, metadata.subunits, "subunit"
        )
        return work

    def _upsert_version(self, DocumentVersion, work, metadata):
        DocumentVersion.objects.select_for_update().filter(
            work=work,
            is_current=True,
        ).exclude(dok_id=metadata.dok_id).update(is_current=False)
        defaults = {
            "version_label": metadata.short_title or "",
            "is_current": True,
            "in_force": True,
            "last_changed_at": self._ensure_aware(metadata.last_updated),
            "last_change_in_force": self._ensure_aware(metadata.last_change_in_force),
            "last_changed_by": metadata.last_changed_by or "",
            "metadata": {"date_in_force": metadata.date_in_force},
        }
        version, _ = DocumentVersion.objects.update_or_create(dok_id=metadata.dok_id, defaults=defaults, work=work)
        return version

    def _sync_sections(self, DocumentSection, version, document: ExtractedDocument):
        DocumentSection.objects.filter(version=version).delete()
        section_lookup: Dict[str, Any] = {}
        for section in document.sections:
            parent = section_lookup.get(section.parent_section_id)
            obj = DocumentSection.objects.create(
                version=version,
                section_id=section.section_id,
                ref_id=section.ref_id,
                heading=section.heading or "",
                html=section.html,
                text=section.text,
                level=section.level,
                order=section.order,
                parent=parent,
            )
            section_lookup[section.section_id] = obj
        return section_lookup

    def _sync_relationships(self, models, work, relationships: List[DocumentRelationship]):
        DocumentWork = models["DocumentWork"]
        DocumentRelationship = models["DocumentRelationship"]
        for rel in relationships:
            target, _ = DocumentWork.objects.get_or_create(
                ref_id=rel.target_ref_id,
                defaults={"document_type": DocumentWork.DocumentType.OTHER},
            )
            relation_type = self.relation_map.get(rel.relation_type, "related")
            DocumentRelationship.objects.update_or_create(
                from_work=work,
                to_work=target,
                relation_type=relation_type,
                defaults={"evidence": rel.description or ""},
            )

    def _assign_legal_areas(self, LegalArea, work, titles: List[str]):
        if not titles:
            return
        area_ids = []
        for title in titles:
            if not title:
                continue
            code = self._slugify(title)
            area, _ = LegalArea.objects.get_or_create(code=code, defaults={"title": title})
            area_ids.append(area.id)
        if area_ids:
            work.legal_areas.set(area_ids)

    def _assign_organizations(self, Organization, RoleModel, work, names: List[str], role: str):
        if not names:
            return
        for name in names:
            if not name:
                continue
            org, _ = Organization.objects.get_or_create(name=name, defaults={"type": role})
            RoleModel.objects.get_or_create(work=work, organization=org, role=role)

    def save_chunks(self, saved: SavedDocument, chunks: List[Chunk], vector_ids: List[str]) -> None:
        if not chunks:
            return
        if len(vector_ids) != len(chunks):
            raise ValueError("Mismatch between chunks and vector IDs.")
        models = self._get_models()
        ChunkRef = models["ChunkRef"]
        DocumentVersion = models["DocumentVersion"]
        version = saved.version
        with transaction.atomic():
            version = DocumentVersion.objects.select_for_update().get(pk=version.pk)
            ChunkRef.objects.filter(version=version).delete()
            for chunk, vector_id in zip(chunks, vector_ids, strict=True):
                chunk.vector_id = vector_id
                section = saved.sections_by_id.get(chunk.section_id)
                ChunkRef.objects.create(
                    version=version,
                    section=section,
                    chunk_id=chunk.chunk_id,
                    vector_store_id=vector_id,
                    order=chunk.order,
                    metadata=chunk.metadata,
                )

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug[:32] or "area"

    def _normalize_document_type(self, DocumentWork, raw_document_type: Optional[str]) -> str:
        normalized = self.work_type_map.get((raw_document_type or "").lower())
        if normalized is not None:
            return normalized
        logger.warning("Unknown document_type=%r, storing as 'other'", raw_document_type)
        return DocumentWork.DocumentType.OTHER

    @staticmethod
    def _first_date(values: List[str]) -> Optional[date]:
        for value in values:
            try:
                return datetime.fromisoformat(value).date()
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _ensure_aware(value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        try:
            from django.utils import timezone
        except Exception:
            return value
        if timezone.is_aware(value):
            return value
        return timezone.make_aware(value)

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        return value


# Separators for user-uploaded documents (contracts, memos, etc.)
DOCUMENT_SEPARATORS = [
    "\n\n\n",  # Major section breaks
    "\n\n",    # Paragraph breaks
    "\n",      # Line breaks
    ". ",      # Sentence boundaries
    ", ",      # Clause boundaries
    " ",       # Word boundaries
    "",        # Character fallback
]


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks respecting natural boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=DOCUMENT_SEPARATORS,
        length_function=len,
    )
    return splitter.split_text(text)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


async def ingest_user_document(
    conversation_id: str,
    document_id: str,
    filename: str,
    text: str,
    embedding_service,
    vector_store: UserDocumentStore,
) -> tuple[int, int, bool]:
    """
    Ingest a user-uploaded document into Weaviate.

    Truncates text to max_extracted_chars if it exceeds the limit.

    Args:
        conversation_id: UUID of the conversation
        document_id: UUID of the UserDocument
        filename: Original filename
        text: Extracted text content
        embedding_service: EmbeddingService or compatible async embedding service
        vector_store: UserDocumentStore instance

    Returns:
        Tuple of (token_count, chunk_count, was_truncated)
    """
    import asyncio

    from config.app_settings import upload_settings

    # Truncate if text exceeds limit
    was_truncated = False
    if len(text) > upload_settings.max_extracted_chars:
        original_len = len(text)
        text = text[:upload_settings.max_extracted_chars]
        was_truncated = True
        logger.warning(
            "Document truncated from %d to %d chars: %s",
            original_len,
            upload_settings.max_extracted_chars,
            filename,
        )

    token_count = count_tokens(text)
    chunks = chunk_text(text)

    if not chunks:
        logger.warning("No chunks generated for document=%s", filename)
        return token_count, 0, was_truncated

    # Generate embeddings
    embeddings = await embedding_service.embed(chunks)

    # Store in Weaviate (sync call wrapped for async context)
    await asyncio.to_thread(
        vector_store.upsert_chunks,
        texts=chunks,
        embeddings=embeddings,
        conversation_id=conversation_id,
        document_id=document_id,
        document_name=filename,
    )

    logger.info(
        "Ingested user document filename=%s tokens=%s chunks=%s conversation=%s truncated=%s",
        filename, token_count, len(chunks), conversation_id, was_truncated,
    )
    return token_count, len(chunks), was_truncated
