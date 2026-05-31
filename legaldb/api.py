from __future__ import annotations

import threading
from typing import Dict

from django.http import JsonResponse
from django.views.decorators.http import require_GET
from pydantic import BaseModel, Field, ValidationError

from config.app_settings import search_settings
from legal_pipeline.retriever import DocumentRetriever, MAX_RETRIEVAL_QUERY_CHARS, normalize_retrieval_query

retriever: DocumentRetriever | None = None
retriever_lock = threading.Lock()

ALLOWED_DOCUMENT_TYPES = {"law", "forskrift"}


class SearchQuery(BaseModel):
    """Validated search query parameters."""

    q: str = Field(min_length=1, max_length=MAX_RETRIEVAL_QUERY_CHARS, description="Lexical query for BM25 search")
    semantic_q: str = Field(default="", max_length=MAX_RETRIEVAL_QUERY_CHARS, description="Semantic query for vector search (defaults to q)")
    alpha: float = Field(default=search_settings.search_alpha, ge=0.0, le=1.0)
    in_force: bool = False
    exclude_ref_ids: list[str] = Field(default_factory=list)


def _parse_per_type_limits(params: dict) -> Dict[str, int]:
    """Parse per-type chunk limits from query params.

    Accepts params like: law_chunks=64, forskrift_chunks=128
    Returns: {"law": 64, "forskrift": 128}
    """
    limits: Dict[str, int] = {}
    for key, value in params.items():
        if not key.endswith("_chunks"):
            continue
        document_type = key.removesuffix("_chunks")
        if document_type not in ALLOWED_DOCUMENT_TYPES:
            raise ValueError(f"Unknown document type: {document_type}")
        limits[document_type] = max(1, min(256, int(value)))
    return limits


def _get_retriever() -> DocumentRetriever:
    global retriever
    with retriever_lock:
        if retriever is None:
            retriever = DocumentRetriever()
        return retriever


@require_GET
async def search_documents(request):
    retriever_instance = _get_retriever()

    # Parse and validate query params with Pydantic
    params = dict(request.GET.items())
    q = normalize_retrieval_query(params.pop("q", "") or "")
    semantic_q = normalize_retrieval_query(params.pop("semantic_q", "") or "")
    exclude_raw = params.pop("exclude_ref_ids", "")

    try:
        query = SearchQuery(
            q=q,
            semantic_q=semantic_q or q,  # Default to lexical query if not provided
            alpha=float(params.pop("alpha", search_settings.search_alpha)),
            in_force=params.pop("in_force", "").lower() in {"1", "true", "yes", "on"},
            exclude_ref_ids=[x.strip() for x in exclude_raw.split(",") if x.strip()],
        )
    except (ValidationError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        per_type = _parse_per_type_limits(params)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    if not per_type:
        return JsonResponse(
            {"error": "Per-type chunk limits required. Provide `<type>_chunks`, e.g. `law_chunks=64`, `forskrift_chunks=64`."},
            status=400,
        )

    chunks = await retriever_instance.aretrieve_by_type(
        lexical_query=query.q,
        semantic_query=query.semantic_q,
        per_type=per_type,
        alpha=query.alpha,
        in_force_only=query.in_force,
        exclude_ref_ids=query.exclude_ref_ids or None,
    )

    payload = {
        "lexical_query": query.q,
        "semantic_query": query.semantic_q,
        "alpha": query.alpha,
        "in_force": query.in_force,
        "per_type": per_type,
        "chunks": [chunk.model_dump() for chunk in chunks],
    }
    return JsonResponse(payload)
