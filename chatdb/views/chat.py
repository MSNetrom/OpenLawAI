from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
import uuid as uuid_mod
from datetime import timedelta
from typing import List, Optional
from uuid import UUID

from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.db import connections, transaction
from django.utils import timezone
from django.utils.text import Truncator
from adrf.views import APIView
from rest_framework import permissions, status
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
from chatdb.models import ChatConversation, ChatMessage, UserDocument
from chatdb.locks import ConversationProcessingLock
from usage.models import UsageCallLedger, UsageRequestSummary
from config.redis_client import get_sync_redis_client, redis_key
from chatdb.serializers import ChatRequestSerializer
from chatdb.views.helpers import (
    _delete_stale_document_transaction,
    _format_sse,
    _persist_messages_transaction,
    _persist_streamed_reply_fallback_transaction,
    _select_llm_overlap,
    _store_uploaded_document,
    DocumentUploadRejected,
)
from config.app_settings import upload_settings

logger = logging.getLogger(__name__)


class ChatAPIView(APIView):
    """
    Chat API — accepts a message, starts processing in the background,
    returns 202 JSON immediately. Events are consumed via GET /events/.
    """
    permission_classes = [permissions.IsAuthenticated]

    async def post(self, request, *args, **kwargs):
        request_started = time.monotonic()
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        conversation_id = serializer.validated_data.get("conversation_id")
        user_id = request.user.id
        message = serializer.validated_data["message"]
        quality_mode = serializer.validated_data.get("quality_mode", "thorough")
        logger.info(
            "chat post start conversation_id=%s user_id=%s quality_mode=%s",
            conversation_id, user_id, quality_mode,
        )

        conversation = await self._get_conversation(conversation_id, request)
        cid = str(conversation.id)
        logger.info("chat post conversation resolved conversation_id=%s", cid)

        # Acquire conversation lock (prevents duplicate processing)
        conversation_lock = ConversationProcessingLock(conversation.id)
        if not await conversation_lock.try_acquire():
            return Response(
                {"conversation_id": cid, "code": "conversation_busy",
                 "detail": "Conversation is already being processed. Please wait or reload."},
                status=409,
            )
        logger.info("chat post lock acquired conversation_id=%s", cid)

        # --- Starter path: persist, set Redis keys, create task ---
        task_started = False
        run_id = uuid_mod.uuid4().hex
        r = get_sync_redis_client()

        try:
            # Persist user message to both UI and LLM channels immediately
            await self._persist_user_message(conversation, message)

            # Initialize Redis keys for this run
            await asyncio.to_thread(r.set, redis_key("chat_active_run", cid), run_id, ex=1800)
            await asyncio.to_thread(r.set, redis_key("chat_text", cid, run_id), b"", ex=1800)

            # Fire-and-forget background task — detach from request context
            # so the ASGI handler's ThreadSensitiveContext teardown doesn't
            # kill the executor used by async Django ORM in the pipeline.
            asyncio.create_task(
                _run_detached(_process_chat(
                    view=self,
                    request_user=request.user,
                    conversation=conversation,
                    message=message,
                    quality_mode=quality_mode,
                    request_started=request_started,
                    conversation_lock=conversation_lock,
                    run_id=run_id,
                )),
                context=contextvars.Context(),
            )
            task_started = True

        except Exception:
            logger.exception("chat post starter path failed conversation_id=%s", cid)
            # Cleanup Redis keys
            await asyncio.to_thread(
                r.delete,
                redis_key("chat_active_run", cid),
                redis_key("chat_text", cid, run_id),
            )
            return Response(
                {"conversation_id": cid, "detail": "Something went wrong. Please try again."},
                status=500,
            )

        finally:
            if not task_started:
                await conversation_lock.release()

        logger.info("chat post accepted conversation_id=%s run_id=%s", cid, run_id)
        return Response({"conversation_id": cid}, status=202)

    async def _persist_user_message(self, conversation: ChatConversation, content: str) -> None:
        """Persist user message to both UI and LLM channels immediately."""
        last_user_msg = await (
            conversation.messages
            .filter(channel="ui", role="user")
            .order_by("-created_at")
            .afirst()
        )
        doc_qs = conversation.documents.all()
        if last_user_msg:
            doc_qs = doc_qs.filter(created_at__gt=last_user_msg.created_at)
        user_docs = [
            {"id": str(doc.id), "filename": doc.filename}
            async for doc in doc_qs
        ]
        metadata = {}
        if user_docs:
            metadata["attached_documents"] = user_docs

        await ChatMessage.objects.abulk_create([
            ChatMessage(
                conversation=conversation, role="user", content=content,
                channel="ui", metadata=metadata,
            ),
            ChatMessage(
                conversation=conversation, role="user", content=content,
                channel="llm",
            ),
        ])
        # Set title from first user message
        if not conversation.title:
            conversation.title = Truncator(content).chars(80)
            await conversation.asave(update_fields=["title", "updated_at"])

    async def _get_conversation(self, conversation_id: Optional[UUID], request) -> ChatConversation:
        user = request.user if request.user.is_authenticated else None
        if conversation_id:
            qs = ChatConversation.objects.filter(deleted_at__isnull=True)
            if user:
                qs = qs.filter(user=user)
            else:
                qs = qs.filter(user__isnull=True)
            try:
                return await qs.aget(pk=conversation_id)
            except ChatConversation.DoesNotExist:
                raise Http404("Conversation not found")
        if user:
            return await ChatConversation.objects.acreate(user=user)
        return await ChatConversation.objects.acreate(user=None)

    async def _build_history(self, conversation: ChatConversation) -> ChatHistory:
        history = ChatHistory()
        async for msg in conversation.messages.filter(channel="ui").order_by("created_at", "id").aiterator():
            history.ui_chat_history_raw.new_message(role=msg.role, content=msg.content)
        async for msg in conversation.messages.filter(channel="llm").order_by("created_at", "id").aiterator():
            history.llm_chat_history_raw.new_message(role=msg.role, content=msg.content)
        # If no LLM messages exist (legacy data), copy UI messages to LLM
        if not history.llm_chat_history_raw.conversation_history:
            for msg in history.ui_chat_history_raw.conversation_history:
                history.llm_chat_history_raw.conversation_history.append(msg.model_copy())
        # Restore metadata (including retrieval/documents) from conversation
        try:
            history.metadata = ChatMetadata.model_validate(conversation.metadata or {})
        except ValidationError:
            logger.exception(
                "Invalid conversation metadata conversation_id=%s; using empty metadata fallback",
                conversation.id,
            )
            history.metadata = ChatMetadata()
        # Merge Django-authoritative fields into persisted user_docs state
        persisted = {doc.id: doc for doc in history.metadata.user_docs.documents}
        merged = []
        async for db_doc in conversation.documents.all():
            doc_id = str(db_doc.id)
            if doc_id in persisted:
                existing = persisted[doc_id]
                existing.status = db_doc.status
                existing.token_count = db_doc.token_count
                existing.chunk_count = db_doc.chunk_count
                existing.weaviate_ingested = db_doc.weaviate_ingested
                merged.append(existing)
            else:
                merged.append(UserDoc(
                    id=doc_id,
                    filename=db_doc.filename,
                    status=db_doc.status,
                    token_count=db_doc.token_count,
                    chunk_count=db_doc.chunk_count,
                    weaviate_ingested=db_doc.weaviate_ingested,
                ))
        history.metadata.user_docs.documents = merged
        # Always set conversation_id for user document retrieval
        history.metadata.conversation_id = str(conversation.id)
        return history

    async def _persist_messages(
        self,
        conversation: ChatConversation,
        new_ui_entries: List[Message],
        full_llm_history: List[Message],
    ) -> None:
        """
        Persist messages:
        - UI: append new entries only
        - LLM: preserve unchanged overlap, delete dropped rows, append new rows
        """
        existing_llm_keys = [
            (message.role, message.content)
            async for message in conversation.messages.filter(channel="llm").order_by("created_at", "id").aiterator()
        ]
        desired_llm_keys = [(entry.role, entry.content) for entry in full_llm_history]
        expected_overlap = _select_llm_overlap(existing_llm_keys, desired_llm_keys)
        await asyncio.to_thread(
            _persist_messages_transaction,
            conversation.id,
            new_ui_entries,
            full_llm_history,
            existing_llm_keys,
            expected_overlap,
        )

    async def _persist_messages_with_recovery(
        self,
        conversation: ChatConversation,
        new_ui_entries: List[Message],
        full_llm_history: List[Message],
        streamed_text: str,
    ) -> None:
        try:
            await self._persist_messages(conversation, new_ui_entries, full_llm_history)
            return
        except Exception:
            logger.exception("Primary message persistence failed conversation_id=%s", conversation.id)

        try:
            await self._persist_messages(conversation, new_ui_entries, full_llm_history)
            return
        except Exception:
            logger.exception("Retry message persistence failed conversation_id=%s", conversation.id)

        assistant_text = streamed_text.strip()
        if not assistant_text:
            for entry in reversed(new_ui_entries):
                if entry.role == ChatMessage.Role.ASSISTANT and entry.content.strip():
                    assistant_text = entry.content.strip()
                    break
        if not assistant_text:
            raise RuntimeError("Unable to recover streamed reply persistence without assistant text")

        await asyncio.to_thread(
            _persist_streamed_reply_fallback_transaction,
            conversation.id,
            assistant_text,
        )
        logger.warning("Recovered streamed reply persistence with assistant-only fallback conversation_id=%s", conversation.id)

    async def _update_conversation_metadata(self, conversation: ChatConversation, history: ChatHistory) -> None:
        conversation.metadata = history.metadata.model_dump()
        await conversation.asave(update_fields=["metadata", "updated_at"])


    async def _record_usage(
        self,
        user,
        conversation: ChatConversation | None,
        request_label: str,
        usage_calls: list[dict],
        *,
        search_performed: bool = False,
        ocr_performed: bool = False,
    ) -> None:
        """Persist token usage telemetry after successful request completion."""
        if not usage_calls:
            return

        total_input_tokens = sum(call["input_tokens"] for call in usage_calls)
        total_output_tokens = sum(call["output_tokens"] for call in usage_calls)

        def _do_transaction():
            with transaction.atomic():
                summary = UsageRequestSummary.objects.create(
                    user=user,
                    conversation=conversation,
                    request_label=request_label,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    search_performed=search_performed,
                    ocr_performed=ocr_performed,
                )
                for call in usage_calls:
                    UsageCallLedger.objects.create(
                        request_summary=summary,
                        model=call["model"],
                        input_tokens=call["input_tokens"],
                        output_tokens=call["output_tokens"],
                    )
            return summary

        summary = await asyncio.to_thread(_do_transaction)
        logger.info(
            "usage recorded user=%s summary_id=%s calls=%d in=%d out=%d",
            user.id, summary.id, len(usage_calls),
            total_input_tokens, total_output_tokens,
        )

    async def _cleanup_stale_documents(self, conversation: ChatConversation) -> None:
        """Remove documents that haven't been referenced recently."""
        now = timezone.now()
        stale_time_threshold = now - timedelta(hours=chat_settings.user_doc_stale_hours)
        message_count = await conversation.messages.filter(channel="ui").acount()
        stale_message_threshold = message_count - chat_settings.user_doc_stale_messages

        stale_docs = []
        async for doc in conversation.documents.aiterator():
            if doc.last_referenced_at is None:
                continue
            is_time_stale = doc.last_referenced_at < stale_time_threshold
            is_message_stale = doc.message_count_at_reference < stale_message_threshold
            if is_time_stale and is_message_stale:
                stale_docs.append(doc)

        if not stale_docs:
            return

        for doc in stale_docs:
            deleted_filename = await asyncio.to_thread(_delete_stale_document_transaction, doc.id)
            if deleted_filename is None:
                continue
            logger.info(
                "Deleted stale document id=%s filename=%s",
                doc.id, deleted_filename,
            )



async def _run_detached(coro):
    """Run a coroutine with its own ThreadSensitiveContext.

    Background tasks created via asyncio.create_task() inherit the ASGI
    request's context.  When the request finishes, its ThreadSensitiveContext
    is torn down — killing the CurrentThreadExecutor that async Django ORM
    methods depend on.  Wrapping in a fresh ThreadSensitiveContext gives the
    background task its own executor that lives as long as the task does.
    """
    async with ThreadSensitiveContext():
        await coro


async def _process_chat(
    *,
    view: ChatAPIView,
    request_user,
    conversation: ChatConversation,
    message: str,
    quality_mode: str,
    request_started: float,
    conversation_lock: ConversationProcessingLock,
    run_id: str,
) -> None:
    """
    Background task: runs the chat pipeline, publishes events to Redis,
    and releases locks when done.
    """
    cid = str(conversation.id)
    user_id = request_user.id
    request_label = message[:50]
    channel = redis_key("chat_events", cid, run_id)
    text_key = redis_key("chat_text", cid, run_id)
    status_key = redis_key("chat_status", cid, run_id)
    result_key = redis_key("chat_result", cid, run_id)
    active_run_key = redis_key("chat_active_run", cid)

    r = get_sync_redis_client()
    history: Optional[ChatHistory] = None
    mode_runs_start = None
    reply_persisted = False
    usage_recorded = False

    async def publish(payload: dict) -> None:
        await asyncio.to_thread(r.publish, channel, json.dumps(payload))

    async def refresh_ttls() -> None:
        refreshed = await conversation_lock.refresh()
        if not refreshed:
            raise RuntimeError(f"process_chat lock refresh failed cid={cid} run_id={run_id}")
        await asyncio.to_thread(r.expire, text_key, 1800)
        await asyncio.to_thread(r.expire, status_key, 1800)
        # Only refresh active_run if it still points to our run
        current = await asyncio.to_thread(r.get, active_run_key)
        if current and current.decode() == run_id:
            await asyncio.to_thread(r.expire, active_run_key, 1800)

    async def publish_heartbeat() -> None:
        await publish({"type": "heartbeat"})

    def request_usage_calls() -> list[dict]:
        return [u.model_dump() for u in history.usage_calls]

    def request_usage_flags() -> tuple[bool, bool]:
        mode_runs = history.metadata.mode_runs
        search_performed = (
            mode_runs.retrieve > mode_runs_start.retrieve
            or mode_runs.user_doc_retrieve > mode_runs_start.user_doc_retrieve
        )
        ocr_performed = mode_runs.process_documents > mode_runs_start.process_documents
        return search_performed, ocr_performed

    async def record_usage_if_needed(*, absorb_failure: bool = False) -> None:
        nonlocal usage_recorded
        if usage_recorded or history is None:
            return
        usage_calls = request_usage_calls()
        search_performed, ocr_performed = request_usage_flags()
        try:
            await view._record_usage(
                request_user, conversation, request_label, usage_calls,
                search_performed=search_performed,
                ocr_performed=ocr_performed,
            )
        except Exception:
            if not absorb_failure:
                raise
            logger.exception(
                "process_chat failed to record usage cid=%s user_id=%s",
                cid, user_id,
            )
            return
        usage_recorded = True

    manager = None
    try:
        history = await view._build_history(conversation)
        mode_runs_start = history.metadata.mode_runs.model_copy()
        prev_ui_len = len(history.ui_chat_history_raw.conversation_history)
        manager = ChatManager()

        # Stream events from ChatManager
        async for event in manager.handle_message_streaming(
            message, history, quality_mode=quality_mode, append_user_message=False,
        ):
            if isinstance(event, StatusEvent):
                logger.info("process_chat status: %s", event.message)
                payload = {"type": "status", "message": event.message, "mode": event.mode}
                await asyncio.to_thread(r.set, status_key, json.dumps(payload), ex=1800)
                await publish(payload)
                await refresh_ttls()

            elif isinstance(event, ChunkEvent):
                text_bytes = event.text.encode("utf-8")
                byte_offset = await asyncio.to_thread(r.append, text_key, text_bytes)
                payload = {"type": "chunk", "text": event.text, "offset": byte_offset}
                await publish(payload)
                await refresh_ttls()

            elif isinstance(event, HeartbeatEvent):
                await publish_heartbeat()
                await refresh_ttls()

            elif isinstance(event, ErrorEvent):
                logger.info("process_chat error event: %s", event.detail)
                error_payload = {"type": "error", "detail": _sanitize_error(event.detail)}
                await asyncio.to_thread(r.set, result_key, json.dumps(error_payload), ex=120)
                await publish(error_payload)
                return

        # --- Post-processing: persist, record usage, cleanup ---
        await publish_heartbeat()
        await refresh_ttls()

        new_ui_entries = history.ui_chat_history_raw.conversation_history[prev_ui_len:]
        full_llm_history = history.llm_chat_history_raw.conversation_history
        raw_streamed_text = await asyncio.to_thread(r.get, text_key)
        streamed_text = raw_streamed_text.decode("utf-8") if raw_streamed_text else ""
        await view._persist_messages_with_recovery(
            conversation,
            new_ui_entries,
            full_llm_history,
            streamed_text,
        )
        reply_persisted = True

        await publish_heartbeat()
        await refresh_ttls()
        await view._update_conversation_metadata(conversation, history)

        await publish_heartbeat()
        await refresh_ttls()
        await record_usage_if_needed(absorb_failure=True)

        await publish_heartbeat()
        await refresh_ttls()
        try:
            await view._cleanup_stale_documents(conversation)
        except Exception as e:
            logger.warning("Document cleanup failed: %s", e)

        await publish_heartbeat()
        await refresh_ttls()

        # Build done payload
        done_payload = {
            "type": "done",
            "status": "completed",
            "conversation_id": cid,
        }
        await asyncio.to_thread(r.set, result_key, json.dumps(done_payload), ex=120)
        await publish(done_payload)

        elapsed_ms = int((time.monotonic() - request_started) * 1000)
        logger.info("process_chat complete ms=%s cid=%s", elapsed_ms, cid)

    except Exception:
        logger.exception("process_chat error cid=%s user_id=%s", cid, user_id)
        if reply_persisted:
            try:
                await record_usage_if_needed()
            except Exception:
                logger.exception("process_chat failed to record usage cid=%s user_id=%s", cid, user_id)
        error_payload = {"type": "error", "detail": "Something went wrong. Please try again."}
        try:
            await asyncio.to_thread(r.set, result_key, json.dumps(error_payload), ex=120)
            await publish(error_payload)
        except Exception:
            logger.exception("process_chat failed to publish error cid=%s", cid)

    finally:
        if manager is not None:
            try:
                await manager.aclose()
            except Exception:
                logger.warning("process_chat failed to close manager cid=%s", cid)
        # Grace window for reconnecting clients
        try:
            await asyncio.to_thread(r.expire, text_key, 120)
            await asyncio.to_thread(r.expire, status_key, 120)
            # Only expire active_run if still ours
            current = await asyncio.to_thread(r.get, active_run_key)
            if current and current.decode() == run_id:
                await asyncio.to_thread(r.expire, active_run_key, 120)
        except Exception:
            logger.warning("process_chat grace-window expire failed cid=%s", cid)
        try:
            await conversation_lock.release()
        finally:
            await sync_to_async(connections.close_all, thread_sensitive=True)()

