from __future__ import annotations

from rest_framework import serializers

from chatdb.models import ChatConversation, ChatMessage, UserDocument


class ChatConversationSerializer(serializers.ModelSerializer):
    """Serializer for ChatConversation with denormalized message fields."""

    class Meta:
        model = ChatConversation
        fields = [
            "id",
            "title",
            "metadata",
            "created_at",
            "updated_at",
            "last_message",
            "message_count",
        ]
        read_only_fields = fields


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ["id", "role", "content", "metadata", "created_at"]
        read_only_fields = fields


class ChatRequestSerializer(serializers.Serializer):
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    message = serializers.CharField(min_length=1, max_length=40000)
    quality_mode = serializers.ChoiceField(
        choices=["fast", "thorough"],
        default="thorough",
        required=False,
    )


class UserDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserDocument
        fields = [
            "id",
            "filename",
            "content_type",
            "status",
            "page_count",
            "token_count",
            "chunk_count",
            "created_at",
            "metadata",
        ]
        read_only_fields = fields
