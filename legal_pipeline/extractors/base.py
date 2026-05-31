"""Base protocol for document extractors.

To add support for a new jurisdiction or legal data source, implement the
DocumentExtractor protocol and register it in the ingest command.

Example implementations:
- LovdataExtractor: Norwegian laws and regulations (Lovdata HTML format)

Future extractors could cover:
- UK legislation (legislation.gov.uk XML format)
- US Code (USC XML/HTML format)
- EU legislation (EUR-Lex XML format)
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from legal_pipeline.data_structures import ExtractedDocument


@runtime_checkable
class DocumentExtractor(Protocol):
    """Protocol for extracting structured legal documents from raw source files."""

    def parse_file(self, path: Path) -> ExtractedDocument:
        """Parse a single source file into a structured ExtractedDocument."""
        ...

    def parse_html(self, html: str, *, source_path: Path | None = None) -> ExtractedDocument:
        """Parse HTML/XML content into a structured ExtractedDocument."""
        ...
