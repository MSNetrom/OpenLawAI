from django.contrib import admin

from chatdb.models import ChatConversation


@admin.register(ChatConversation)
class ChatConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "message_count", "deleted_at", "created_at", "updated_at")
    list_filter = ("deleted_at", "created_at")
    search_fields = ("id", "title", "user__username")
    ordering = ("-updated_at",)
