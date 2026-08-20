from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from pathlib import PurePosixPath
from uuid import UUID

from django.http import Http404, StreamingHttpResponse
from adrf.views import APIView
from rest_framework import permissions, status
from rest_framework.response import Response

from chatdb.models import ChatConversation, GeneratedDocument, UserDocument
from chatdb.serializers import UserDocumentSerializer
from chatdb.views.helpers import DocumentUploadRejected, _store_uploaded_document
from config.app_settings import upload_settings
from legal_pipeline.document_extractor import count_pages
import fitz as _fitz

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_FILE_SIZE = 50 * 1024 * 1024  # 50MB for images
MAX_UPLOAD_FILENAME_LENGTH = 255

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


def _sanitize_uploaded_filename(filename: str) -> str:
    raw = (filename or "").replace("\\", "/")
    cleaned = str(PurePosixPath(raw).name)
    cleaned = "".join(ch for ch in cleaned if ch >= " " and ch != "\x7f")
    cleaned = cleaned.strip().strip(".")
    if not cleaned:
        cleaned = "upload"
    if len(cleaned) <= MAX_UPLOAD_FILENAME_LENGTH:
        return cleaned
    if "." in cleaned:
        stem, ext = cleaned.rsplit(".", 1)
        ext = f".{ext[:32]}"
        max_stem_length = max(1, MAX_UPLOAD_FILENAME_LENGTH - len(ext))
        return f"{stem[:max_stem_length]}{ext}"
    return cleaned[:MAX_UPLOAD_FILENAME_LENGTH]


def _looks_like_text_upload(file_bytes: bytes) -> bool:
    sample = file_bytes[:4096]
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    decoded = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            decoded = sample.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        return False
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\n\r\t")
    return printable / max(len(decoded), 1) >= 0.9


def _detect_upload_content_type(file_bytes: bytes) -> str | None:
    if file_bytes.startswith(b"%PDF-"):
        return "application/pdf"
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(file_bytes) >= 12 and file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        return "image/webp"
    if file_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if file_bytes.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
                if "word/document.xml" in archive.namelist():
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        except zipfile.BadZipFile:
            return None
    if _looks_like_text_upload(file_bytes):
        return "text/plain"
    return None


def _upload_content_type_matches(file_bytes: bytes, claimed_content_type: str) -> bool:
    detected = _detect_upload_content_type(file_bytes)
    if detected is None:
        return False
    if claimed_content_type in {"text/plain", "text/markdown"}:
        return detected == "text/plain"
    return detected == claimed_content_type


class DocumentUploadView(APIView):
    """Upload a document to a conversation. Extraction happens during chat flow."""

    permission_classes = [permissions.IsAuthenticated]

    async def post(self, request, conversation_id: UUID):
        try:
            conversation = await ChatConversation.objects.filter(
                user=request.user, deleted_at__isnull=True
            ).aget(pk=conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")

        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"detail": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

        content_type = uploaded_file.content_type
        if content_type not in ALLOWED_CONTENT_TYPES:
            return Response(
                {"detail": f"Unsupported file type: {content_type}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if uploaded_file.size > MAX_FILE_SIZE:
            return Response(
                {"detail": f"File too large. Max size is {MAX_FILE_SIZE // (1024*1024)}MB."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if content_type.startswith("image/") and uploaded_file.size > MAX_IMAGE_FILE_SIZE:
            return Response(
                {"detail": f"Image too large. Max size is {MAX_IMAGE_FILE_SIZE // (1024*1024)}MB."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        file_bytes = await asyncio.to_thread(uploaded_file.read)
        if not _upload_content_type_matches(file_bytes, content_type):
            return Response(
                {"detail": "File type does not match content."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        filename = _sanitize_uploaded_filename(uploaded_file.name)

        # Count pages in new document
        try:
            new_page_count = await asyncio.to_thread(count_pages, file_bytes, content_type)
        except (_fitz.FileDataError, RuntimeError):
            return Response(
                {"detail": "Could not read the document."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if new_page_count == 0:
            return Response(
                {"detail": "Could not read the document."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Check pages per document limit
        if new_page_count > upload_settings.max_pages_per_document:
            return Response(
                {"detail": f"Document has too many pages ({new_page_count}). Max {upload_settings.max_pages_per_document} pages per document."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            user_doc_id = await asyncio.to_thread(
                _store_uploaded_document,
                request.user.id,
                conversation.id,
                filename,
                content_type,
                file_bytes,
                new_page_count,
            )
        except DocumentUploadRejected as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        user_doc = await UserDocument.objects.aget(pk=user_doc_id)

        return Response(
            {"document": UserDocumentSerializer(user_doc).data},
            status=status.HTTP_201_CREATED,
        )

    async def get(self, request, conversation_id: UUID):
        """List all documents in a conversation."""
        try:
            conversation = await ChatConversation.objects.filter(
                user=request.user, deleted_at__isnull=True
            ).aget(pk=conversation_id)
        except ChatConversation.DoesNotExist:
            raise Http404("Conversation not found")

        documents = [doc async for doc in conversation.documents.aiterator()]
        return Response({"documents": UserDocumentSerializer(documents, many=True).data})


class GeneratedDocumentDownloadView(APIView):
    """Download a generated document (PDF/DOCX)."""

    permission_classes = [permissions.IsAuthenticated]

    async def get(self, request, document_id: UUID):
        try:
            generated_doc = await GeneratedDocument.objects.select_related("conversation").aget(pk=document_id)
        except GeneratedDocument.DoesNotExist:
            raise Http404("Document not found")

        if generated_doc.conversation.user_id != request.user.id:
            raise Http404("Document not found")

        async def _chunks():
            yield generated_doc.file_data

        response = StreamingHttpResponse(
            _chunks(),
            content_type=generated_doc.content_type,
        )
        response["Content-Disposition"] = f'attachment; filename="{generated_doc.filename}"'
        response["Content-Length"] = len(generated_doc.file_data)

        logger.info(
            "Generated document downloaded user=%s doc=%s filename=%s",
            request.user.id, document_id, generated_doc.filename,
        )

        return response
