from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import AsyncIterator
from uuid import UUID

from asgiref.sync import sync_to_async
from django.http import Http404, StreamingHttpResponse
from adrf.views import APIView
from rest_framework import permissions
from rest_framework.negotiation import BaseContentNegotiation
from rest_framework.response import Response

from chatdb.models import ChatConversation
from config.redis_client import get_async_redis_client, get_sync_redis_client, redis_key
from chatdb.views.helpers import _format_sse, IgnoreClientContentNegotiation

logger = logging.getLogger(__name__)

CHAT_STARTUP_WAIT_SECONDS = 15.0
CHAT_STARTUP_POLL_SECONDS = 0.25


class ConversationEventsAPIView(APIView):
    """
    GET /api/conversations/{id}/events/ — SSE endpoint for live events.

    Discovers the current run_id from Redis, subscribes to the pub/sub channel,
    sends a catch-up snapshot, then streams live events with dead-task detection.
    """
    permission_classes = [permissions.IsAuthenticated]
    content_negotiation_class = IgnoreClientContentNegotiation

    async def _wait_for_active_run(self, conversation_id: str) -> str | None:
        r_sync = get_sync_redis_client()
        deadline = time.monotonic() + CHAT_STARTUP_WAIT_SECONDS
        active_run_key = redis_key("chat_active_run", conversation_id)
        lock_key = redis_key("chat_processing", conversation_id)

        while time.monotonic() < deadline:
            raw_run_id = await asyncio.to_thread(r_sync.get, active_run_key)
            if raw_run_id is not None:
                return raw_run_id.decode()
            if not await asyncio.to_thread(r_sync.exists, lock_key):
                return None
            await asyncio.sleep(CHAT_STARTUP_POLL_SECONDS)
        return None


    async def get(self, request, conversation_id: UUID):
        # Verify ownership
        try:
            conversation = await ChatConversation.objects.filter(
                user=request.user, deleted_at__isnull=True,
            ).aget(pk=conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")

        cid = str(conversation.id)
        r_sync = get_sync_redis_client()
        try:
            raw_run_id = await asyncio.to_thread(r_sync.get, redis_key("chat_active_run", cid))
            if raw_run_id is None:
                lock_exists = await asyncio.to_thread(r_sync.exists, redis_key("chat_processing", cid))
                if not lock_exists:
                    return self._terminal_sse({"type": "done", "status": "not_processing"})
                response = StreamingHttpResponse(
                    self._startup_event_stream(cid, request.user.id),
                    content_type="text/event-stream",
                )
                response["Cache-Control"] = "no-cache"
                response["X-Accel-Buffering"] = "no"
                return response

            run_id = raw_run_id.decode()
            result_key = redis_key("chat_result", cid, run_id)

            raw_result = await asyncio.to_thread(r_sync.get, result_key)
            if raw_result is not None:
                return self._terminal_sse(json.loads(raw_result))

            if not await asyncio.to_thread(r_sync.exists, redis_key("chat_processing", cid)):
                raw_result = await asyncio.to_thread(r_sync.get, result_key)
                if raw_result is not None:
                    return self._terminal_sse(json.loads(raw_result))
                return self._terminal_sse({"type": "done", "status": "not_processing"})

            channel_name = redis_key("chat_events", cid, run_id)
            text_key = redis_key("chat_text", cid, run_id)
            status_key = redis_key("chat_status", cid, run_id)

            response = StreamingHttpResponse(
                self._event_stream(cid, run_id, channel_name, text_key, status_key, result_key),
                content_type="text/event-stream",
            )
            response["Cache-Control"] = "no-cache"
            response["X-Accel-Buffering"] = "no"
            return response
        finally:
            pass

    def _terminal_sse(self, payload: dict) -> StreamingHttpResponse:
        """Return an SSE response with a single terminal event."""
        event_type = payload.get("type", "done")
        body = _format_sse(event_type, json.dumps(payload))

        async def gen():
            yield body

        resp = StreamingHttpResponse(gen(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp

    async def _startup_event_stream(
        self,
        cid: str,
        user_id: int,
    ) -> AsyncIterator[str]:
        r_sync = get_sync_redis_client()
        try:
            yield _format_sse("status", json.dumps({
                "type": "status",
                "message": "Starter behandling...",
                "mode": None,
            }))
            run_id = await self._wait_for_active_run(cid)
            if run_id is None:
                lock_exists = await asyncio.to_thread(r_sync.exists, redis_key("chat_processing", cid))
                terminal_status = "stalled" if lock_exists else "not_processing"
                yield _format_sse("done", json.dumps({"type": "done", "status": terminal_status}))
                return

            channel_name = redis_key("chat_events", cid, run_id)
            text_key = redis_key("chat_text", cid, run_id)
            status_key = redis_key("chat_status", cid, run_id)
            result_key = redis_key("chat_result", cid, run_id)
            async for chunk in self._event_stream(
                cid,
                run_id,
                channel_name,
                text_key,
                status_key,
                result_key,
            ):
                yield chunk
        except Exception:
            logger.exception("Startup SSE stream failed conversation_id=%s user_id=%s", cid, user_id)
            yield _format_sse("error", json.dumps({"type": "error", "detail": "Something went wrong. Please try again."}))

    async def _event_stream(
        self, cid: str, run_id: str,
        channel_name: str, text_key: str, status_key: str,
        result_key: str,
    ) -> AsyncIterator[str]:
        """Async generator: subscribe → snapshot → stream → dead-task detect."""
        r_async = get_async_redis_client()
        pubsub = r_async.pubsub()
        r_sync = get_sync_redis_client()
        cleanup = AsyncExitStack()
        cleanup.push_async_callback(pubsub.aclose)
        cleanup.push_async_callback(pubsub.unsubscribe, channel_name)

        try:
            # 1. Subscribe FIRST
            await pubsub.subscribe(channel_name)

            # 2. Snapshot text + status
            raw_bytes = await asyncio.to_thread(r_sync.get, text_key)
            snapshot_bytes = raw_bytes if raw_bytes is not None else b""
            snapshot_offset = len(snapshot_bytes)
            raw_status = await asyncio.to_thread(r_sync.get, status_key)

            # 3. Re-check result (terminal race window)
            raw_result = await asyncio.to_thread(r_sync.get, result_key)
            if raw_result is not None:
                terminal = json.loads(raw_result)
                if snapshot_bytes:
                    yield _format_sse("chunk", json.dumps({
                        "text": snapshot_bytes.decode("utf-8"), "is_catchup": True,
                    }))
                yield _format_sse(terminal.get("type", "done"), json.dumps(terminal))
                return

            # 4. Send catch-up status (actual last status from pipeline)
            if raw_status is not None:
                yield _format_sse("status", raw_status.decode("utf-8"))

            # 5. Send catch-up chunk
            if snapshot_bytes:
                yield _format_sse("chunk", json.dumps({
                    "text": snapshot_bytes.decode("utf-8"), "is_catchup": True,
                }))

            # 5. Stream live events with dead-task detection
            consecutive_timeouts = 0
            while True:
                try:
                    msg = await asyncio.wait_for(pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=15.0,
                    ), timeout=15.0)
                except asyncio.TimeoutError:
                    msg = None
                if msg is not None and msg["type"] == "message":
                    consecutive_timeouts = 0
                    try:
                        data = json.loads(msg["data"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    event_type = data.get("type", "")

                    if event_type == "heartbeat":
                        yield ": heartbeat\n\n"
                        continue

                    if event_type == "chunk":
                        offset = data.get("offset", 0)
                        if offset <= snapshot_offset:
                            continue
                        # Forward as live chunk (no is_catchup flag)
                        yield _format_sse("chunk", json.dumps({"text": data["text"]}))
                        continue

                    if event_type == "status":
                        yield _format_sse("status", json.dumps(data))
                        continue

                    if event_type in ("done", "error"):
                        yield _format_sse(event_type, json.dumps(data))
                        return

                    if event_type == "notice":
                        yield _format_sse("notice", json.dumps(data))
                        continue

                    # Unknown event — forward as-is
                    yield _format_sse(event_type, json.dumps(data))
                    continue

                # Timeout path — send SSE keepalive comment
                yield ": heartbeat\n\n"

                # Dead-task detection checks
                raw_result = await asyncio.to_thread(r_sync.get, result_key)
                if raw_result is not None:
                    terminal = json.loads(raw_result)
                    yield _format_sse(terminal.get("type", "done"), json.dumps(terminal))
                    return

                lock_exists = await asyncio.to_thread(r_sync.exists, redis_key("chat_processing", cid))
                if not lock_exists:
                    yield _format_sse("done", json.dumps({"type": "done", "status": "not_processing"}))
                    return

                consecutive_timeouts += 1
                if consecutive_timeouts >= 4:
                    yield _format_sse("done", json.dumps({"type": "done", "status": "stalled"}))
                    return

        finally:
            await cleanup.aclose()

