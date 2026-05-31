"""Consolidated Weaviate client using the official v4 SDK.

This module provides a unified interface for all Weaviate operations:
- LegalChunkStore: search and upsert for legal document chunks
- UserDocumentStore: search, upsert, and delete for user uploads
"""

from __future__ import annotations

import atexit
import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.collections import Collection

logger = logging.getLogger(__name__)

# Module-level singleton for connection reuse
_weaviate_client: weaviate.WeaviateClient | None = None
_weaviate_client_lock = threading.Lock()


def _normalize_failed_batch_ids(failed_objects: Sequence[Any]) -> set[str]:
    failed_ids: set[str] = set()
    for failed in failed_objects:
        original_uuid = getattr(failed, "original_uuid", None)
        if original_uuid is None:
            continue
        failed_ids.add(str(original_uuid))
    return failed_ids


def _cleanup_partial_batch_successes(
    collection: Collection,
    *,
    attempted_uuids: Sequence[uuid.UUID],
    failed_objects: Sequence[Any],
    label: str,
) -> None:
    failed_ids = _normalize_failed_batch_ids(failed_objects)
    successful_uuids = [chunk_uuid for chunk_uuid in attempted_uuids if str(chunk_uuid) not in failed_ids]
    if not successful_uuids:
        return

    deleted = 0
    for chunk_uuid in successful_uuids:
        try:
            collection.data.delete_by_id(chunk_uuid)
            deleted += 1
        except Exception:
            logger.exception("Failed cleaning up partial %s batch uuid=%s", label, chunk_uuid)
    logger.warning("Cleaned up %d partial %s vector(s) after batch failure", deleted, label)


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Get or create a singleton Weaviate client connection."""
    global _weaviate_client
    with _weaviate_client_lock:
        if _weaviate_client is not None and _weaviate_client.is_connected():
            return _weaviate_client
        if _weaviate_client is not None:
            _weaviate_client.close()
            _weaviate_client = None

        endpoint = os.environ["WEAVIATE_ENDPOINT"]
        # Parse endpoint to extract host and port
        clean = endpoint.replace("http://", "").replace("https://", "")
        host = clean.split(":")[0]
        port = int(clean.split(":")[-1].split("/")[0]) if ":" in clean else 8080
        secure = endpoint.startswith("https://")
        grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

        _weaviate_client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=secure,
        )
        return _weaviate_client


def close_weaviate_client() -> None:
    """Close the singleton Weaviate client connection."""
    global _weaviate_client
    with _weaviate_client_lock:
        if _weaviate_client is not None:
            _weaviate_client.close()
            _weaviate_client = None


atexit.register(close_weaviate_client)


# ---------------------------------------------------------------------------
# LegalChunk Collection
# ---------------------------------------------------------------------------

LEGAL_CHUNK_COLLECTION = "LegalChunk"

LEGAL_CHUNK_PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="ref_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="dok_id", data_type=DataType.TEXT),
    Property(name="section_id", data_type=DataType.TEXT),
    Property(name="document_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="in_force", data_type=DataType.BOOL),
    Property(name="legal_source", data_type=DataType.TEXT),
    Property(name="title", data_type=DataType.TEXT),
    Property(name="short_title", data_type=DataType.TEXT),
    Property(name="ministries", data_type=DataType.TEXT),
    Property(name="legal_areas", data_type=DataType.TEXT),
    Property(name="applies_to", data_type=DataType.TEXT),
    Property(name="authority_refs", data_type=DataType.TEXT),
    Property(name="date_in_force", data_type=DataType.TEXT),
    Property(name="date_of_publication", data_type=DataType.TEXT),
    Property(name="misc_information", data_type=DataType.TEXT),
    Property(name="metadata", data_type=DataType.TEXT),
]


def _ensure_legal_chunk_collection(client: weaviate.WeaviateClient) -> Collection:
    """Ensure the LegalChunk collection exists with the correct schema."""
    if client.collections.exists(LEGAL_CHUNK_COLLECTION):
        return client.collections.get(LEGAL_CHUNK_COLLECTION)

    return client.collections.create(
        name=LEGAL_CHUNK_COLLECTION,
        description="Chunks of legal documents",
        vector_config=Configure.Vectors.self_provided(),
        properties=LEGAL_CHUNK_PROPERTIES,
    )


@dataclass(slots=True)
class ChunkHit:
    """Raw chunk hit returned from Weaviate before hydration."""

    vector_id: str
    ref_id: str
    section_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


def _build_legal_filter(
    document_types: Optional[Sequence[str]] = None,
    in_force_only: bool = False,
    exclude_ref_ids: Optional[Sequence[str]] = None,
) -> Optional[Filter]:
    """Build a Weaviate filter for legal chunk queries."""
    conditions: List[Filter] = []

    if document_types:
        normalized = [t.strip() for t in document_types if t and t.strip()]
        if len(normalized) == 1:
            conditions.append(Filter.by_property("document_type").equal(normalized[0]))
        elif len(normalized) > 1:
            type_conds = [Filter.by_property("document_type").equal(t) for t in normalized]
            combined = type_conds[0]
            for tc in type_conds[1:]:
                combined = combined | tc
            conditions.append(combined)

    if in_force_only:
        conditions.append(Filter.by_property("in_force").equal(True))

    if exclude_ref_ids:
        for ref_id in exclude_ref_ids:
            conditions.append(Filter.by_property("ref_id").not_equal(ref_id))

    if not conditions:
        return None

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined & c
    return combined


class LegalChunkStore:
    """Weaviate store for legal document chunks.

    Provides hybrid search and batch upsert operations.
    """

    def __init__(self, client: weaviate.WeaviateClient | None = None) -> None:
        self._client = client
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            client = self._client or get_weaviate_client()
            self._client = client
            self._collection = _ensure_legal_chunk_collection(client)
        return self._collection

    async def _get_collection_async(self) -> Collection:
        return await asyncio.to_thread(self._get_collection)

    async def hybrid_search(
        self,
        query: str,
        vector: Optional[Sequence[float]] = None,
        limit: int = 32,
        alpha: float = 0.5,
        document_types: Optional[Sequence[str]] = None,
        in_force_only: bool = False,
        exclude_ref_ids: Optional[Sequence[str]] = None,
    ) -> List[ChunkHit]:
        """Perform hybrid search on legal chunks."""
        filters = _build_legal_filter(
            document_types=document_types,
            in_force_only=in_force_only,
            exclude_ref_ids=exclude_ref_ids,
        )

        query_kwargs = {
            "query": query,
            "alpha": alpha,
            "limit": limit,
            "filters": filters,
            "return_metadata": MetadataQuery(score=True, distance=True),
        }
        if vector is not None:
            query_kwargs["vector"] = list(vector)
        collection = await self._get_collection_async()
        response = await asyncio.to_thread(collection.query.hybrid, **query_kwargs)

        results: List[ChunkHit] = []
        for obj in response.objects:
            props = obj.properties
            vector_id = str(obj.uuid)

            score = 0.0
            if obj.metadata and obj.metadata.score is not None:
                score = obj.metadata.score
            elif obj.metadata and obj.metadata.distance is not None:
                score = 1 - obj.metadata.distance

            metadata_raw = props.get("metadata")
            metadata: Dict[str, Any] = {}
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw

            metadata.setdefault("document_type", props.get("document_type"))
            metadata.setdefault("in_force", props.get("in_force"))

            results.append(
                ChunkHit(
                    vector_id=vector_id,
                    ref_id=props.get("ref_id") or "",
                    section_id=props.get("section_id") or "",
                    text=props.get("text") or "",
                    metadata=metadata,
                    score=float(score),
                )
            )
        return results

    def upsert_chunks(
        self,
        chunks: List[Any],  # List[Chunk] from data_structures
    ) -> List[str]:
        """Batch upsert chunks with their embeddings. Returns UUIDs.
        
        Uses deterministic UUIDs based on chunk_id for idempotent upserts.
        """
        if not chunks:
            return []

        # Generate deterministic UUIDs based on chunk_id for idempotency
        # This ensures the same chunk always gets the same UUID
        NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace
        chunk_uuids = [uuid.uuid5(NAMESPACE, chunk.chunk_id) for chunk in chunks]
        collection = self._get_collection()

        with collection.batch.dynamic() as batch:
            for chunk, chunk_uuid in zip(chunks, chunk_uuids):
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.chunk_id} is missing embedding.")
                properties = {
                    "text": chunk.text,
                    "ref_id": chunk.metadata["ref_id"],
                    "dok_id": chunk.metadata["dok_id"],
                    "section_id": chunk.section_id,
                    "document_type": chunk.metadata["document_type"],
                    "in_force": True,
                    "legal_source": chunk.metadata["legal_source"],
                    "title": chunk.metadata["title"],
                    "short_title": chunk.metadata["short_title"],
                    "ministries": chunk.metadata["ministries"],
                    "legal_areas": chunk.metadata["legal_areas"],
                    "applies_to": chunk.metadata["applies_to"],
                    "authority_refs": chunk.metadata["authority_refs"],
                    "date_in_force": chunk.metadata["date_in_force"],
                    "date_of_publication": chunk.metadata["date_of_publication"],
                    "misc_information": chunk.metadata["misc_information"],
                    "metadata": json.dumps(chunk.metadata),
                }
                batch.add_object(
                    properties=properties, 
                    vector=chunk.embedding,
                    uuid=chunk_uuid,
                )

        # Check for batch errors
        # batch.number_errors is available on context var, but failed_objects is on collection.batch
        if batch.number_errors > 0:
            failed = collection.batch.failed_objects
            error_msgs = [f"{f.original_uuid}: {f.message}" for f in failed[:5]]
            logger.error(f"Batch insert had {len(failed)} errors: {error_msgs}")
            _cleanup_partial_batch_successes(
                collection,
                attempted_uuids=chunk_uuids,
                failed_objects=failed,
                label="legal chunk",
            )
            raise ValueError(f"Failed to insert {len(failed)} chunks: {error_msgs}")

        return [str(u) for u in chunk_uuids]

    def delete_by_vector_ids(self, vector_ids: Sequence[str]) -> None:
        """Delete legal chunk vectors by UUID."""
        collection = self._get_collection()
        deleted = 0
        for vector_id in vector_ids:
            try:
                collection.data.delete_by_id(uuid.UUID(vector_id))
                deleted += 1
            except ValueError:
                logger.warning("Skipping malformed legal chunk vector_id=%r during delete", vector_id)
        logger.info("Deleted %d legal chunk vectors", deleted)


# ---------------------------------------------------------------------------
# UserDocumentChunk Collection
# ---------------------------------------------------------------------------

USER_DOCUMENT_COLLECTION = "UserDocumentChunk"

USER_DOCUMENT_PROPERTIES = [
    Property(name="text", data_type=DataType.TEXT),
    Property(name="conversation_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="document_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
    Property(name="document_name", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="metadata", data_type=DataType.TEXT),
]


def _ensure_user_document_collection(client: weaviate.WeaviateClient) -> Collection:
    """Ensure the UserDocumentChunk collection exists."""
    if client.collections.exists(USER_DOCUMENT_COLLECTION):
        return client.collections.get(USER_DOCUMENT_COLLECTION)

    return client.collections.create(
        name=USER_DOCUMENT_COLLECTION,
        description="Chunks from user-uploaded documents",
        vector_config=Configure.Vectors.self_provided(),
        properties=USER_DOCUMENT_PROPERTIES,
    )


@dataclass(slots=True)
class UserDocumentChunkHit:
    """Chunk hit from user-uploaded documents."""

    vector_id: str
    text: str
    document_id: str
    document_name: str
    chunk_index: int
    score: float
    metadata: Dict[str, Any]


class UserDocumentStore:
    """Weaviate store for user-uploaded document chunks.

    Provides hybrid search, batch upsert, and deletion operations.
    """

    def __init__(self, client: weaviate.WeaviateClient | None = None) -> None:
        self._client = client
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            client = self._client or get_weaviate_client()
            self._client = client
            self._collection = _ensure_user_document_collection(client)
        return self._collection

    async def _get_collection_async(self) -> Collection:
        return await asyncio.to_thread(self._get_collection)

    async def hybrid_search(
        self,
        query: str,
        conversation_id: str,
        vector: Optional[Sequence[float]] = None,
        limit: int = 20,
        alpha: float = 0.5,
    ) -> List[UserDocumentChunkHit]:
        """Search user documents filtered by conversation_id."""
        filters = Filter.by_property("conversation_id").equal(conversation_id)

        query_kwargs = {
            "query": query,
            "alpha": alpha,
            "limit": limit,
            "filters": filters,
            "return_metadata": MetadataQuery(score=True, distance=True),
        }
        if vector is not None:
            query_kwargs["vector"] = list(vector)
        collection = await self._get_collection_async()
        response = await asyncio.to_thread(collection.query.hybrid, **query_kwargs)

        results: List[UserDocumentChunkHit] = []
        for obj in response.objects:
            props = obj.properties
            vector_id = str(obj.uuid)

            score = 0.0
            if obj.metadata and obj.metadata.score is not None:
                score = obj.metadata.score
            elif obj.metadata and obj.metadata.distance is not None:
                score = 1 - obj.metadata.distance

            metadata_raw = props.get("metadata")
            metadata: Dict[str, Any] = {}
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    pass
            elif isinstance(metadata_raw, dict):
                metadata = metadata_raw

            results.append(
                UserDocumentChunkHit(
                    vector_id=vector_id,
                    text=props.get("text") or "",
                    document_id=props.get("document_id") or "",
                    document_name=props.get("document_name") or "",
                    chunk_index=props.get("chunk_index") or 0,
                    score=float(score),
                    metadata=metadata,
                )
            )
        return results

    def upsert_chunks(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        conversation_id: str,
        document_id: str,
        document_name: str,
    ) -> List[str]:
        """Batch upsert user document chunks. Returns UUIDs."""
        if not texts:
            return []

        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        chunk_uuids = [uuid.uuid5(namespace, f"{document_id}:{i}") for i in range(len(texts))]
        ids: List[str] = []
        collection = self._get_collection()
        with collection.batch.dynamic() as batch:
            for i, (text, embedding, chunk_uuid) in enumerate(zip(texts, embeddings, chunk_uuids, strict=True)):
                properties = {
                    "text": text,
                    "conversation_id": conversation_id,
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_index": i,
                    "metadata": json.dumps({
                        "conversation_id": conversation_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "chunk_index": i,
                    }),
                }
                batch.add_object(properties=properties, vector=embedding, uuid=chunk_uuid)
                ids.append(str(chunk_uuid))

        # Check for batch errors
        if batch.number_errors > 0:
            failed = collection.batch.failed_objects
            error_msgs = [f"{f.original_uuid}: {f.message}" for f in failed[:5]]
            logger.error(f"Batch insert had {len(failed)} errors: {error_msgs}")
            _cleanup_partial_batch_successes(
                collection,
                attempted_uuids=chunk_uuids,
                failed_objects=failed,
                label="user document chunk",
            )
            raise ValueError(f"Failed to insert {len(failed)} user document chunks: {error_msgs}")

        logger.info(
            "Indexed %d user document chunks for conversation=%s document=%s",
            len(ids),
            conversation_id,
            document_name,
        )
        return ids

    def delete_by_document(self, document_id: str) -> None:
        """Delete all chunks for a specific document."""
        self._get_collection().data.delete_many(
            where=Filter.by_property("document_id").equal(document_id)
        )
        logger.info("Deleted chunks for document_id=%s", document_id)

    def delete_by_vector_ids(self, vector_ids: Sequence[str]) -> None:
        """Delete user document chunk vectors by UUID."""
        collection = self._get_collection()
        deleted = 0
        for vector_id in vector_ids:
            try:
                collection.data.delete_by_id(uuid.UUID(vector_id))
                deleted += 1
            except ValueError:
                logger.warning("Skipping malformed user document vector_id=%r during delete", vector_id)
        logger.info("Deleted %d user document vectors", deleted)

    def delete_by_conversation(self, conversation_id: str) -> None:
        """Delete all chunks for a specific conversation."""
        self._get_collection().data.delete_many(
            where=Filter.by_property("conversation_id").equal(conversation_id)
        )
        logger.info("Deleted chunks for conversation_id=%s", conversation_id)
