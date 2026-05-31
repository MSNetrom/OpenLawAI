from __future__ import annotations

from django.conf import settings
from django.db import models


class UsageRequestSummary(models.Model):
    """Summary of token usage for a single user request (chat message)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="usage_summaries",
    )
    conversation = models.ForeignKey(
        "chatdb.ChatConversation",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="usage_summaries",
    )
    request_label = models.CharField(max_length=50)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    search_performed = models.BooleanField(default=False)
    ocr_performed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.user_id}: {self.request_label[:20]}... ({self.input_tokens}in/{self.output_tokens}out)"


class UsageCallLedger(models.Model):
    """Individual LLM call within a request, for detailed tracking."""

    request_summary = models.ForeignKey(
        UsageRequestSummary,
        on_delete=models.PROTECT,
        related_name="calls",
    )
    model = models.CharField(max_length=64)
    input_tokens = models.PositiveIntegerField()
    output_tokens = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.model}: {self.input_tokens}in/{self.output_tokens}out"
