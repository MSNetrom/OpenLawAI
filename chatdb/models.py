from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class ChatConversationQuerySet(models.QuerySet):
    def active(self):
        return self.filter(deleted_at__isnull=True)


class ActiveChatConversationManager(models.Manager):
    def get_queryset(self):
        return ChatConversationQuerySet(self.model, using=self._db).active()


class ChatConversation(models.Model):
    objects = ActiveChatConversationManager()
    all_objects = ChatConversationQuerySet.as_manager()

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_conversations",
    )
    title = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    # Denormalized fields for quick list queries (source of truth is LangGraph checkpointer)
    last_message = models.TextField(blank=True, default="")
    message_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-updated_at"]
        default_manager_name = "objects"
        base_manager_name = "all_objects"
        indexes = [
            models.Index(fields=["user", "deleted_at", "updated_at"], name="chatconv_user_del_upd_idx"),
        ]

    def __str__(self) -> str:
        return self.title or f"Conversation {self.id}"


class ChatMessage(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"

    class Channel(models.TextChoices):
        UI = "ui", "UI"
        LLM = "llm", "LLM"

    conversation = models.ForeignKey(
        ChatConversation,
        related_name="messages",
        on_delete=models.CASCADE,
    )
    role = models.CharField(max_length=16, choices=Role.choices)
    channel = models.CharField(max_length=8, choices=Channel.choices, default="ui")
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["conversation", "channel", "created_at"], name="chatmsg_conv_chan_created_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:40]}"


class UserDocument(models.Model):
    """User-uploaded document attached to a conversation."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending extraction"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        ChatConversation,
        related_name="documents",
        on_delete=models.CASCADE,
    )
    filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=64)
    # Raw file bytes - stored until extraction is complete
    file_data = models.BinaryField(null=True, blank=True)
    # Processing status
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
    )
    # Extracted content (populated after processing)
    extracted_text = models.TextField(blank=True, default="")
    page_count = models.IntegerField(default=0)
    token_count = models.IntegerField(default=0)
    chunk_count = models.IntegerField(default=0)
    weaviate_ingested = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    # Usage tracking for cleanup
    last_referenced_at = models.DateTimeField(null=True, blank=True)
    message_count_at_reference = models.IntegerField(default=0)
    # Additional metadata (e.g., was_truncated, page_count)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["conversation", "status"], name="userdoc_conv_status_idx"),
        ]

    def __str__(self) -> str:
        if self.status == self.Status.READY:
            return f"{self.filename} ({self.token_count} tokens)"
        return f"{self.filename} ({self.status})"


class GeneratedDocument(models.Model):
    """LLM-generated document (PDF/DOCX) attached to a conversation."""

    class Format(models.TextChoices):
        PDF = "pdf", "PDF"
        DOCX = "docx", "DOCX"
        MD = "md", "Markdown"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        ChatConversation,
        related_name="generated_documents",
        on_delete=models.CASCADE,
    )
    filename = models.CharField(max_length=255)
    format = models.CharField(max_length=10, choices=Format.choices)
    markdown_source = models.TextField()
    file_data = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.filename} ({self.format})"

    @property
    def content_type(self) -> str:
        content_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "md": "text/markdown",
        }
        return content_types[self.format]
