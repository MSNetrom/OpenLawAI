from django.conf import settings
from django.contrib import admin
from django.urls import path

from legaldb.api import search_documents
from legaldb.views import chat_ui, dev_chat_ui
from chatdb.views import (
    ChatAPIView,
    ConversationDetailAPIView,
    ConversationDeveloperAPIView,
    ConversationEventsAPIView,
    ConversationListAPIView,
    DocumentUploadView,
    GeneratedDocumentDownloadView,
)
from accounts.views import (
    LoginAPIView,
    LogoutAPIView,
    RegisterAPIView,
    SessionAPIView,
)
from usage.views import UsageHistoryAPIView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/search/", search_documents, name="search_documents"),
    path("api/chat/", ChatAPIView.as_view(), name="chat_api"),
    path("api/conversations/", ConversationListAPIView.as_view(), name="conversation_list_api"),
    path("api/conversations/<uuid:conversation_id>/", ConversationDetailAPIView.as_view(), name="conversation_detail_api"),
    path("api/conversations/<uuid:conversation_id>/events/", ConversationEventsAPIView.as_view(), name="conversation_events_api"),
    path("api/conversations/<uuid:conversation_id>/documents/", DocumentUploadView.as_view(), name="document_upload_api"),
    path("api/documents/<uuid:document_id>/", GeneratedDocumentDownloadView.as_view(), name="generated_document_download_api"),

    # Auth
    path("api/auth/register/", RegisterAPIView.as_view(), name="register_api"),
    path("api/auth/login/", LoginAPIView.as_view(), name="login_api"),
    path("api/auth/logout/", LogoutAPIView.as_view(), name="logout_api"),
    path("api/auth/session/", SessionAPIView.as_view(), name="session_api"),

    # Usage
    path("api/usage/", UsageHistoryAPIView.as_view(), name="usage_history_api"),

    path("", chat_ui, name="chat_ui"),
]

if settings.DEBUG:
    urlpatterns += [
        path("dev/", dev_chat_ui, name="chat_ui_dev"),
        path("api/dev/conversations/<uuid:conversation_id>/", ConversationDeveloperAPIView.as_view(), name="conversation_dev_detail_api"),
    ]
