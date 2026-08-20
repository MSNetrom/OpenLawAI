from __future__ import annotations

import logging

from adrf.views import APIView
from django.db.models import Prefetch
from rest_framework import permissions
from rest_framework.response import Response

from usage.models import UsageCallLedger, UsageRequestSummary

logger = logging.getLogger(__name__)


class UsageHistoryAPIView(APIView):
    """Get recent token usage history for the authenticated user."""

    permission_classes = [permissions.IsAuthenticated]

    async def get(self, request):
        try:
            limit = int(request.query_params.get("limit", 20))
            offset = int(request.query_params.get("offset", 0))
        except (ValueError, TypeError):
            return Response({"detail": "Invalid limit or offset."}, status=400)
        if limit < 1 or offset < 0:
            return Response({"detail": "limit must be >= 1 and offset must be >= 0."}, status=400)
        limit = min(limit, 50)

        qs = (
            UsageRequestSummary.objects.filter(user=request.user)
            .order_by("-created_at")
            .prefetch_related(
                Prefetch("calls", queryset=UsageCallLedger.objects.order_by("created_at"))
            )[offset : offset + limit + 1]
        )
        summaries = []
        has_more = False
        async for summary in qs:
            if len(summaries) >= limit:
                has_more = True
                break
            # Aggregate calls by model
            calls_by_model: dict[str, dict] = {}
            async for call in summary.calls.all():
                if call.model not in calls_by_model:
                    calls_by_model[call.model] = {
                        "model": call.model,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                calls_by_model[call.model]["input_tokens"] += call.input_tokens
                calls_by_model[call.model]["output_tokens"] += call.output_tokens
            calls = list(calls_by_model.values())
            services = []
            if summary.search_performed:
                services.append("search")
            if summary.ocr_performed:
                services.append("ocr")
            summaries.append({
                "id": summary.id,
                "request_label": summary.request_label,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "created_at": summary.created_at.isoformat(),
                "calls": calls,
                "services": services,
            })
        return Response({"usage": summaries, "has_more": has_more})
