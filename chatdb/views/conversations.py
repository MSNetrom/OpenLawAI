from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from asgiref.sync import sync_to_async
from django.db.models import Count, Q
from django.http import Http404
from adrf.views import APIView
from rest_framework import permissions, status
from rest_framework.response import Response

from chatdb.models import ChatConversation, ChatMessage
from chatdb.serializers import ChatConversationSerializer, ChatMessageSerializer
from chatdb.views.helpers import (
    MAX_CONVERSATION_DETAIL_LIMIT,
    MAX_CONVERSATION_LIST_LIMIT,
    _public_conversation_metadata,
    _soft_delete_conversation,
)

logger = logging.getLogger(__name__)


class ConversationListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    async def get(self, request):
        try:
            limit = int(request.query_params.get("limit", 20))
            offset = int(request.query_params.get("offset", 0))
        except (ValueError, TypeError):
            return Response({"detail": "Invalid limit or offset."}, status=status.HTTP_400_BAD_REQUEST)
        if limit < 1 or offset < 0 or limit > MAX_CONVERSATION_LIST_LIMIT:
            return Response(
                {"detail": f"limit must be between 1 and {MAX_CONVERSATION_LIST_LIMIT}, and offset must be >= 0."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        conversations_qs = (
            ChatConversation.objects
            .filter(user=request.user, deleted_at__isnull=True)
            .annotate(
                ui_message_count=Count(
                    "messages",
                    filter=Q(messages__channel=ChatMessage.Channel.UI),
                    distinct=True,
                ),
                document_count=Count("documents", distinct=True),
            )
            .filter(Q(ui_message_count__gt=0) | Q(document_count__gt=0))
            .order_by("-updated_at", "-id")
        )

        total = await conversations_qs.acount()
        conversations = [c async for c in conversations_qs[offset:offset + limit].aiterator()]
        conversations_data = ChatConversationSerializer(conversations, many=True).data
        for convo in conversations_data:
            convo["metadata"] = _public_conversation_metadata(convo["metadata"])
        return Response({
            "conversations": conversations_data,
            "has_more": offset + len(conversations) < total,
        })

    async def post(self, request):
        """Create a new empty conversation."""
        conversation = await ChatConversation.objects.acreate(user=request.user)
        logger.info("Created new conversation id=%s user=%s", conversation.id, request.user.id)
        conversation_data = ChatConversationSerializer(conversation).data
        conversation_data["metadata"] = _public_conversation_metadata(conversation.metadata or {})
        return Response(
            {"conversation": conversation_data},
            status=status.HTTP_201_CREATED,
        )


class ConversationDetailAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    async def get(self, request, conversation_id: UUID):
        try:
            limit = int(request.query_params.get("limit", MAX_CONVERSATION_DETAIL_LIMIT))
            offset = int(request.query_params.get("offset", 0))
        except (ValueError, TypeError):
            return Response({"detail": "Invalid limit or offset."}, status=status.HTTP_400_BAD_REQUEST)
        if limit < 1 or offset < 0 or limit > MAX_CONVERSATION_DETAIL_LIMIT:
            return Response(
                {"detail": f"limit must be between 1 and {MAX_CONVERSATION_DETAIL_LIMIT}, and offset must be >= 0."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            conversation = await ChatConversation.objects.filter(
                user=request.user, deleted_at__isnull=True
            ).aget(pk=conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")
        
        convo_data = ChatConversationSerializer(conversation).data
        convo_data["metadata"] = _public_conversation_metadata(conversation.metadata or {})
        ui_messages_qs = conversation.messages.filter(channel="ui").order_by("-created_at")
        total_messages = await ui_messages_qs.acount()
        ui_messages_desc = [msg async for msg in ui_messages_qs[offset:offset + limit].aiterator()]
        ui_messages = list(reversed(ui_messages_desc))
        
        # SECURITY: Do NOT include llm_history - it contains system prompts
        # LLM history is stored server-side and restored via _build_history() during chat
        messages = ChatMessageSerializer(ui_messages, many=True).data
        metadata = _public_conversation_metadata(conversation.metadata or {})
        return Response({
            "conversation": convo_data,
            "messages": messages,
            "metadata": metadata,
            "has_more_messages": offset + len(ui_messages_desc) < total_messages,
            "next_offset": offset + len(ui_messages_desc) if offset + len(ui_messages_desc) < total_messages else None,
        })

    async def patch(self, request, conversation_id: UUID):
        if not isinstance(request.data, dict):
            return Response({"detail": "Invalid request body."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            conversation = await ChatConversation.objects.filter(
                user=request.user, deleted_at__isnull=True
            ).aget(pk=conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")

        title = request.data.get("title")
        update_fields = ["updated_at"]

        if title is not None:
            if not isinstance(title, str) or not title.strip():
                return Response({"detail": "Title cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)
            conversation.title = title.strip()[:255]
            update_fields.append("title")

        await conversation.asave(update_fields=update_fields)
        conversation_data = ChatConversationSerializer(conversation).data
        conversation_data["metadata"] = _public_conversation_metadata(conversation.metadata or {})
        return Response({"conversation": conversation_data})

    async def delete(self, request, conversation_id: UUID):
        try:
            await asyncio.to_thread(_soft_delete_conversation, request.user.id, conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")
        return Response({"deleted": True})


class ConversationDeveloperAPIView(APIView):
    """Developer view that returns ALL messages including LLM-only ones."""
    permission_classes = [permissions.IsAdminUser]

    async def get(self, request, conversation_id: UUID):
        try:
            conversation = await ChatConversation.objects.filter(
                deleted_at__isnull=True
            ).aget(pk=conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")
        
        convo_data = ChatConversationSerializer(conversation).data
        ui_messages = [msg async for msg in conversation.messages.filter(channel="ui").aiterator()]
        llm_messages = [msg async for msg in conversation.messages.filter(channel="llm").aiterator()]
        ui_messages_data = ChatMessageSerializer(ui_messages, many=True).data
        
        # Fallback: if no LLM messages, use UI messages (legacy data)
        if llm_messages:
            llm_history = [{"role": msg.role, "content": msg.content} for msg in llm_messages]
        else:
            llm_history = [{"role": msg.role, "content": msg.content} for msg in ui_messages]
        
        metadata = conversation.metadata or {}
        data = {
            "conversation": convo_data,
            "ui_messages": ui_messages_data,
            "llm_history": llm_history,
            "metadata": metadata,
        }
        return Response(data)




ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}
