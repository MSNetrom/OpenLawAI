from __future__ import annotations

import asyncio
import contextvars
import io
import json
import logging
import time
import uuid as uuid_mod
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import PurePosixPath
from typing import AsyncIterator, List, Optional
from uuid import UUID
import zipfile

from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import connections, transaction
from django.db.models import Count, Q, Sum
from django.http import Http404, StreamingHttpResponse
from django.utils import timezone
from django.utils.text import Truncator
from adrf.views import APIView
from rest_framework import permissions, status
from rest_framework.negotiation import BaseContentNegotiation
from rest_framework.response import Response
from pydantic import ValidationError

from chat_manager import ChatManager, settings as chat_settings
from agents.models import (
    ChatHistory,
    ChatMetadata,
    Message,
    StatusEvent,
    ChunkEvent,
    ErrorEvent,
    HeartbeatEvent,
    UserDoc,
)
from chatdb.models import ChatConversation, ChatMessage, GeneratedDocument, UserDocument
from chatdb.locks import ConversationProcessingLock
from usage.models import UsageCallLedger, UsageRequestSummary
from config.redis_client import get_async_redis_client, get_sync_redis_client, redis_key
from chatdb.serializers import (
    ChatConversationSerializer,
    ChatMessageSerializer,
    ChatRequestSerializer,
    UserDocumentSerializer,
)
from legal_pipeline.weaviate_client import UserDocumentStore
import fitz as _fitz
from legal_pipeline.document_extractor import count_pages
from config.app_settings import upload_settings

MAX_CONVERSATION_LIST_LIMIT = 100
MAX_CONVERSATION_DETAIL_LIMIT = 500


class IgnoreClientContentNegotiation(BaseContentNegotiation):
    """
    Content negotiation class that ignores the client's Accept header.
    Used for SSE endpoints where we always return text/event-stream.
    """
    def select_parser(self, request, parsers):
        return parsers[0]

    def select_renderer(self, request, renderers, format_suffix=None):
        return (renderers[0], renderers[0].media_type)

User = get_user_model()
logger = logging.getLogger(__name__)
CHAT_STARTUP_WAIT_SECONDS = 15.0
CHAT_STARTUP_POLL_SECONDS = 0.25



def _format_sse(event: str, data: str) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {data}\n\n"


def _sanitize_error(detail: str) -> str:
    """Pass through the error detail to the user."""
    return detail


def _public_conversation_metadata(_metadata: dict) -> dict:
    """Public API never exposes internal conversation metadata."""
    return {}


def _delete_document_vectors_after_commit(document_id: str) -> None:
    try:
        UserDocumentStore().delete_by_document(document_id)
    except Exception:
        logger.exception("Failed to delete Weaviate vectors for document_id=%s after commit", document_id)


def _delete_conversation_vectors_after_commit(conversation_id: UUID) -> None:
    try:
        UserDocumentStore().delete_by_conversation(str(conversation_id))
    except Exception:
        logger.exception("Failed to delete Weaviate vectors for conversation_id=%s after commit", conversation_id)


def _delete_stale_document_transaction(document_id: UUID) -> str | None:
    """Delete a stale document in the database and clean up vectors after commit."""
    with transaction.atomic():
        try:
            doc = UserDocument.objects.select_for_update().get(pk=document_id)
        except UserDocument.DoesNotExist:
            return None

        filename = doc.filename
        doc.delete()
        transaction.on_commit(
            lambda document_id=str(document_id): _delete_document_vectors_after_commit(document_id)
        )
        return filename


def _persist_streamed_reply_fallback_transaction(
    conversation_id: UUID,
    assistant_text: str,
) -> None:
    if not assistant_text.strip():
        raise ValueError("assistant_text must be non-empty for fallback persistence")

    with transaction.atomic():
        conversation = ChatConversation.objects.select_for_update().get(pk=conversation_id)
        last_ui = (
            conversation.messages.filter(channel="ui").order_by("-created_at", "-id").first()
        )
        if (
            last_ui
            and last_ui.role == ChatMessage.Role.ASSISTANT
            and last_ui.content == assistant_text
        ):
            return

        conversation.messages.bulk_create(
            [
                ChatMessage(
                    conversation=conversation,
                    role=ChatMessage.Role.ASSISTANT,
                    content=assistant_text,
                    channel=ChatMessage.Channel.UI,
                ),
                ChatMessage(
                    conversation=conversation,
                    role=ChatMessage.Role.ASSISTANT,
                    content=assistant_text,
                    channel=ChatMessage.Channel.LLM,
                ),
            ]
        )
        conversation.last_message = assistant_text
        conversation.message_count = conversation.messages.filter(channel=ChatMessage.Channel.UI).count()
        update_fields = ["last_message", "message_count", "updated_at"]
        if not conversation.title:
            conversation.title = Truncator(assistant_text).chars(80)
            update_fields.append("title")
        conversation.save(update_fields=update_fields)


class DocumentUploadRejected(Exception):
    """Raised when an upload violates per-conversation limits."""


def _select_llm_overlap(
    existing_llm_keys: List[tuple[str, str]],
    desired_llm_keys: List[tuple[str, str]],
) -> tuple[int, int]:
    """Find the optimal overlap between current and desired LLM history.

    Fast paths cover the common append-only and front-pruned cases. The fallback
    preserves the existing semantics for rarer re-synchronization scenarios.
    """
    if not existing_llm_keys or not desired_llm_keys:
        return len(existing_llm_keys), 0

    existing_len = len(existing_llm_keys)
    desired_len = len(desired_llm_keys)

    # Common case: desired history extends the current stored history.
    prefix_overlap = min(existing_len, desired_len)
    if existing_llm_keys[:prefix_overlap] == desired_llm_keys[:prefix_overlap]:
        return 0, prefix_overlap

    # Common case: oldest stored rows were pruned and the desired history starts
    # with a suffix of what is already stored.
    max_suffix_overlap = min(existing_len, desired_len)
    for overlap in range(max_suffix_overlap, 0, -1):
        if existing_llm_keys[-overlap:] == desired_llm_keys[:overlap]:
            return existing_len - overlap, overlap

    best_start = existing_len
    best_overlap = 0
    best_writes = existing_len + desired_len
    first_desired = desired_llm_keys[0]
    candidate_starts = [
        idx
        for idx, key in enumerate(existing_llm_keys)
        if key == first_desired
    ]
    candidate_starts.append(existing_len)

    for start in candidate_starts:
        max_overlap = min(existing_len - start, desired_len)
        overlap = 0
        while overlap < max_overlap and existing_llm_keys[start + overlap] == desired_llm_keys[overlap]:
            overlap += 1
        if overlap != max_overlap:
            continue
        writes = start + (existing_len - start - overlap) + (desired_len - overlap)
        if writes < best_writes:
            best_start = start
            best_overlap = overlap
            best_writes = writes

    return best_start, best_overlap


def _persist_messages_transaction(
    conversation_id: UUID,
    new_ui_entries: List[Message],
    full_llm_history: List[Message],
    expected_existing_llm_keys: List[tuple[str, str]] | None = None,
    expected_overlap: tuple[int, int] | None = None,
) -> None:
    with transaction.atomic():
        conversation = ChatConversation.objects.select_for_update().get(pk=conversation_id)
        existing_llm_messages = list(
            conversation.messages.filter(channel="llm").order_by("created_at", "id")
        )
        existing_llm_keys = [(message.role, message.content) for message in existing_llm_messages]
        desired_llm_keys = [(entry.role, entry.content) for entry in full_llm_history]
        if expected_existing_llm_keys == existing_llm_keys and expected_overlap is not None:
            best_start, best_overlap = expected_overlap
        else:
            best_start, best_overlap = _select_llm_overlap(existing_llm_keys, desired_llm_keys)

        delete_llm_ids = [
            message.id
            for message in (
                existing_llm_messages[:best_start]
                + existing_llm_messages[best_start + best_overlap:]
            )
        ]
        if delete_llm_ids:
            conversation.messages.filter(id__in=delete_llm_ids).delete()

        last_user_msg = (
            conversation.messages
            .filter(channel="ui", role="user")
            .order_by("-created_at")
            .first()
        )
        doc_qs = conversation.documents.all()
        if last_user_msg:
            doc_qs = doc_qs.filter(created_at__gt=last_user_msg.created_at)
        user_docs = [
            {"id": str(doc.id), "filename": doc.filename}
            for doc in doc_qs
        ]

        objs = []
        for entry in new_ui_entries:
            metadata = {}
            if entry.role == "user" and user_docs:
                metadata["attached_documents"] = user_docs
            objs.append(ChatMessage(
                conversation=conversation,
                role=entry.role,
                content=entry.content,
                channel="ui",
                metadata=metadata,
            ))
        for entry in full_llm_history[best_overlap:]:
            objs.append(ChatMessage(
                conversation=conversation,
                role=entry.role,
                content=entry.content,
                channel="llm",
            ))
        if not objs:
            return

        created = ChatMessage.objects.bulk_create(objs)
        if new_ui_entries:
            conversation.last_message = new_ui_entries[-1].content
            conversation.message_count = conversation.messages.filter(channel=ChatMessage.Channel.UI).count()
            update_fields = ["last_message", "message_count", "updated_at"]
            if not conversation.title:
                for message in created:
                    if message.channel == ChatMessage.Channel.UI and message.role == ChatMessage.Role.USER and message.content:
                        conversation.title = Truncator(message.content).chars(80)
                        update_fields.append("title")
                        break
            conversation.save(update_fields=update_fields)
            return
        if conversation.title:
            return
        for message in created:
            if message.channel == ChatMessage.Channel.UI and message.role == ChatMessage.Role.USER and message.content:
                conversation.title = Truncator(message.content).chars(80)
                conversation.save(update_fields=["title", "updated_at"])
                return


def _store_uploaded_document(
    user_id: int,
    conversation_id: UUID,
    filename: str,
    content_type: str,
    file_bytes: bytes,
    new_page_count: int,
) -> UUID:
    existing_ready_document_id: str | None = None
    with transaction.atomic():
        conversation = ChatConversation.objects.select_for_update().get(
            pk=conversation_id,
            user_id=user_id,
            deleted_at__isnull=True,
        )
        existing_doc = UserDocument.objects.filter(
            conversation=conversation,
            filename=filename,
        ).first()

        existing_count = UserDocument.objects.filter(conversation=conversation).count()
        effective_count = existing_count - (1 if existing_doc else 0)
        if effective_count >= upload_settings.max_docs_per_conversation:
            raise DocumentUploadRejected(
                f"Maks {upload_settings.max_docs_per_conversation} dokumenter per samtale."
            )

        existing_pages = UserDocument.objects.filter(conversation=conversation).aggregate(total=Sum("page_count"))["total"] or 0
        replaced_pages = existing_doc.page_count if existing_doc else 0
        total_pages = existing_pages - replaced_pages + new_page_count
        if total_pages > upload_settings.max_total_pages_per_conversation:
            raise DocumentUploadRejected(
                f"Total page count ({total_pages}) exceeds the limit of {upload_settings.max_total_pages_per_conversation} pages per conversation."
            )

        if existing_doc:
            if existing_doc.status == UserDocument.Status.READY:
                existing_ready_document_id = str(existing_doc.id)
            existing_doc.delete()
            logger.info("Replaced existing document filename=%s", filename)

        user_doc = UserDocument.objects.create(
            conversation=conversation,
            filename=filename,
            content_type=content_type,
            file_data=file_bytes,
            page_count=new_page_count,
            status=UserDocument.Status.PENDING,
            last_referenced_at=timezone.now(),
            message_count_at_reference=0,
        )
        conversation.save(update_fields=["updated_at"])
        if existing_ready_document_id is not None:
            transaction.on_commit(
                lambda document_id=existing_ready_document_id: _delete_document_vectors_after_commit(document_id)
            )

    logger.info(
        "Document uploaded (pending) conversation=%s document=%s filename=%s bytes=%s pages=%s",
        conversation_id, user_doc.id, filename, len(file_bytes), new_page_count,
    )
    return user_doc.id


def _soft_delete_conversation(user_id: int, conversation_id: UUID) -> None:
    with transaction.atomic():
        conversation = ChatConversation.all_objects.select_for_update().get(
            pk=conversation_id,
            user_id=user_id,
            deleted_at__isnull=True,
        )
        has_documents = UserDocument.objects.filter(conversation=conversation).exists()
        conversation.deleted_at = timezone.now()
        conversation.save(update_fields=["deleted_at", "updated_at"])
        if has_documents:
            transaction.on_commit(
                lambda: _delete_conversation_vectors_after_commit(conversation_id)
            )


