"""Document generation service using Pandoc for markdown conversion."""
from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

default_url_fetcher = None

DocumentFormat = Literal["pdf", "docx", "md"]
PANDOC_TIMEOUT_SECONDS = 300


class UnsafeDocumentResourceError(ValueError):
    """Raised when generated markdown tries to load external resources during rendering."""


_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(\s*(?!data:)[^)]+\)", re.IGNORECASE)
_MARKDOWN_REFERENCE_PATTERN = re.compile(r"^\[[^\]]+\]:\s*(?!data:)[^\s]+", re.IGNORECASE | re.MULTILINE)
_HTML_RESOURCE_PATTERN = re.compile(
    r"<(?:img|audio|video|source|track|iframe|embed|link|script)\b[^>]*(?:src|href)\s*=\s*[\"']?(?!data:)[^\"'>\s]+",
    re.IGNORECASE,
)
_BLOCKED_HTML_TAG_PATTERN = re.compile(r"<(?:object|svg|meta)\b", re.IGNORECASE)
_CSS_URL_PATTERN = re.compile(r"url\(\s*[\"']?(?!data:)[^)]+\)", re.IGNORECASE)
_CSS_IMPORT_PATTERN = re.compile(r"@import\s+(?![\"']?data:)", re.IGNORECASE)
_YAML_FRONTMATTER_PATTERN = re.compile(r"\A(?:\ufeff)?---\s*\n.*?\n---\s*(?:\n|$)", re.DOTALL)


def _reject_fetchable_resources(markdown: str) -> None:
    if _YAML_FRONTMATTER_PATTERN.match(markdown):
        raise UnsafeDocumentResourceError("Generated markdown contains disallowed YAML front matter.")
    blocked_tag = _BLOCKED_HTML_TAG_PATTERN.search(markdown)
    if blocked_tag:
        raise UnsafeDocumentResourceError(
            f"Generated markdown contains disallowed fetchable resource tag: {blocked_tag.group(0)}"
        )
    for pattern in (
        _MARKDOWN_IMAGE_PATTERN,
        _MARKDOWN_REFERENCE_PATTERN,
        _HTML_RESOURCE_PATTERN,
        _CSS_URL_PATTERN,
        _CSS_IMPORT_PATTERN,
    ):
        match = pattern.search(markdown)
        if match:
            raise UnsafeDocumentResourceError(
                f"Generated markdown contains disallowed fetchable resource reference: {match.group(0)[:160]}"
            )


def _locked_pdf_url_fetcher(url: str) -> dict:
    global default_url_fetcher
    if url.startswith("data:"):
        if default_url_fetcher is None:
            from weasyprint import default_url_fetcher as weasyprint_default_url_fetcher

            default_url_fetcher = weasyprint_default_url_fetcher
        return default_url_fetcher(url)
    raise UnsafeDocumentResourceError(f"PDF rendering attempted to fetch a blocked resource: {url}")


def _run_pandoc(cmd: list[str]) -> None:
    logger.info("Running pandoc: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=PANDOC_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        logger.error("Pandoc stderr: %s", result.stderr)
        raise RuntimeError(f"Pandoc conversion failed: {result.stderr}")


def markdown_to_format(markdown: str, output_format: DocumentFormat) -> bytes:
    """Convert markdown to the specified format using Pandoc.

    Args:
        markdown: The markdown content to convert.
        output_format: Target format ("pdf", "docx", or "md").

    Returns:
        The converted document as bytes.
    """
    if output_format == "md":
        return markdown.encode("utf-8")

    _reject_fetchable_resources(markdown)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.md"
        input_path.write_text(markdown, encoding="utf-8")
        if output_format == "pdf":
            from weasyprint import HTML

            output_path = Path(tmpdir) / "output.html"
            _run_pandoc([
                "pandoc",
                str(input_path),
                "-o",
                str(output_path),
                "--standalone",
                "-t",
                "html5",
            ])
            html = output_path.read_text(encoding="utf-8")
            return HTML(
                string=html,
                base_url=str(Path(tmpdir)),
                url_fetcher=_locked_pdf_url_fetcher,
            ).write_pdf()

        output_path = Path(tmpdir) / f"output.{output_format}"
        _run_pandoc([
            "pandoc",
            str(input_path),
            "-o",
            str(output_path),
            "--standalone",
        ])
        return output_path.read_bytes()


def generate_filename(title: str, output_format: DocumentFormat) -> str:
    """Generate a safe filename from title and format."""
    # Remove/replace unsafe characters
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title.strip()[:50] or "document"
    return f"{safe_title}.{output_format}"
