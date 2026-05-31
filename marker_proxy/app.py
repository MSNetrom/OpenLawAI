"""Marker proxy API that accepts file uploads and returns markdown."""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

logger = logging.getLogger(__name__)

app = FastAPI()

_MARKER_MODELS = None
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024

TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".xml", ".json", ".csv"}


def _load_marker_models() -> dict:
    global _MARKER_MODELS
    if _MARKER_MODELS is None:
        logger.info("marker models loading")
        _MARKER_MODELS = create_model_dict()
        logger.info("marker models loaded")
    return _MARKER_MODELS


def _extract_text_file(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("text", b"", 0, 1, "Unsupported text encoding")


def _extract_with_marker(file_bytes: bytes) -> str:
    converter = PdfConverter(artifact_dict=_load_marker_models())
    rendered = converter(io.BytesIO(file_bytes))
    text, _metadata, _images = text_from_rendered(rendered)
    return text


def _extract_document_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return _extract_text_file(file_bytes)
    return _extract_with_marker(file_bytes)


def _validated_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    return filename


async def _read_limited_upload(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large.")
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


@app.post("/marker")
async def convert(file: UploadFile = File(...)) -> dict:
    file_bytes = await _read_limited_upload(file)
    filename = _validated_filename(file.filename)
    text = await asyncio.to_thread(_extract_document_text, file_bytes, filename)
    logger.info("extracted filename=%s chars=%s", filename, len(text))
    return {"markdown": text}
