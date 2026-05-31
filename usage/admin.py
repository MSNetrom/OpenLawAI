from django.contrib import admin

from usage.models import UsageCallLedger, UsageRequestSummary


@admin.register(UsageRequestSummary)
class UsageRequestSummaryAdmin(admin.ModelAdmin):
    list_display = ("user", "request_label", "input_tokens", "output_tokens", "created_at")
    list_filter = ("created_at",)
    search_fields = ("user__username", "request_label")
    ordering = ("-created_at",)


@admin.register(UsageCallLedger)
class UsageCallLedgerAdmin(admin.ModelAdmin):
    list_display = ("request_summary", "model", "input_tokens", "output_tokens", "created_at")
    list_filter = ("model", "created_at")
    ordering = ("-created_at",)
