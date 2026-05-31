"""Document extraction via Marker API.

Calls the marker_server HTTP endpoint for document extraction.
Plain text formats are decoded locally without API calls.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Protocol

import fitz  # PyMuPDF, installed via marker-pdf
import httpx

logger = logging.getLogger(__name__)


def _decode_plain_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8-sig", file_bytes, 0, len(file_bytes), "Unsupported text encoding")


def count_pages(file_bytes: bytes, content_type: str) -> int:
    """Count pages in a document without full text extraction.

    Args:
        file_bytes: Raw file bytes
        content_type: MIME type of the file

    Returns:
        Number of pages (1 for images/text, actual count for PDFs)
    """
    if content_type == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            return len(doc)
        finally:
            doc.close()

    # Images count as 1 page
    if content_type.startswith("image/"):
        return 1

    # Text files and DOCX count as 1 page
    return 1

MARKER_API_URL = os.environ.get("MARKER_API_URL", "http://localhost:8003")
MARKER_TIMEOUT = float(os.environ.get("MARKER_TIMEOUT_SECONDS", "300"))


class TextExtractor(Protocol):
    """Protocol for extracting text from user-uploaded documents (PDF, DOCX, images)."""

    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from document bytes. Returns markdown."""
        ...


class MarkerExtractor:
    """Extraction via Marker HTTP API.
    
    Handles PDFs, images, DOCX via the Marker server.
    Plain text formats are decoded locally.
    """

    def __init__(self, api_url: str = MARKER_API_URL, timeout: float = MARKER_TIMEOUT) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def extract(self, file_bytes: bytes, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()

        # Plain text - decode locally
        if ext in (".txt", ".md", ".html", ".htm", ".xml", ".json", ".csv"):
            text = _decode_plain_text(file_bytes)
            logger.info("extracted text filename=%s chars=%s", filename, len(text))
            return text

        # Everything else - send to Marker API
        return self._extract_via_api(file_bytes, filename)

    def _extract_via_api(self, file_bytes: bytes, filename: str) -> str:
        url = f"{self.api_url}/marker"
        files = {"file": (filename, io.BytesIO(file_bytes))}

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, files=files)
            response.raise_for_status()

        result = response.json()
        text = result.get("markdown", result.get("text", ""))
        logger.info("extracted via marker api filename=%s chars=%s", filename, len(text))
        return text


def get_extractor() -> TextExtractor:
    """Get the document extractor."""
    return MarkerExtractor()
