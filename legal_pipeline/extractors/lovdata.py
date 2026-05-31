from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

if __package__ is None or __package__ == "":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from bs4 import BeautifulSoup, Tag  # type: ignore
from dotenv import load_dotenv

from legal_pipeline.data_structures import (
    DocumentMetadata,
    DocumentRelationship,
    DocumentSection,
    ExtractedDocument,
)

load_dotenv()


class LovdataExtractor:
    """
    Parses Lovdata HTML files (Norwegian laws and regulations) into structured
    dataclasses that downstream modules can work with.

    Implements the DocumentExtractor protocol.
    """

    def __init__(self, default_legal_source: str = "NL") -> None:
        self.default_legal_source = default_legal_source

    def parse_file(self, path: Path) -> ExtractedDocument:
        html = path.read_text(encoding="utf-8")
        return self.parse_html(html, source_path=path)

    def parse_html(self, html: str, source_path: Optional[Path] = None) -> ExtractedDocument:
        soup = BeautifulSoup(html, "html.parser")
        header = soup.select_one("header.documentHeader")
        if header is None:
            raise ValueError("Could not locate <header class='documentHeader'> block")

        metadata = self._parse_metadata(header)
        main = soup.select_one("main.documentBody")
        sections = self._parse_sections(main, metadata.ref_id) if main else []
        relationships = self._parse_relationships(header, main, metadata.ref_id)

        metadata = replace(metadata, misc_information=self._extract_misc_information(header))
        return ExtractedDocument(metadata=metadata, sections=sections, relationships=relationships)

    def _parse_metadata(self, header: Tag) -> DocumentMetadata:
        field_map = self._collect_dt_dd_pairs(header)

        ref_id = self._extract_text(field_map.get("refid"))
        dok_id = self._extract_text(field_map.get("dokid"))
        legacy_id = self._extract_text(field_map.get("legacyID"))
        title = self._extract_text(field_map.get("title"))
        short_title = self._extract_text(field_map.get("titleShort"))
        document_type, legal_source = self._infer_types(ref_id, dok_id)

        date_in_force = self._extract_list(field_map.get("dateInForce"))
        date_of_publication = self._parse_iso_datetime(self._extract_text(field_map.get("dateOfPublication")))
        last_updated = self._parse_iso_datetime(self._extract_text(field_map.get("lastupdated")))
        last_change_in_force = self._parse_iso_datetime(self._extract_text(field_map.get("lastChangeInForce")))

        return DocumentMetadata(
            ref_id=ref_id,
            dok_id=dok_id,
            legacy_id=legacy_id,
            title=title,
            short_title=short_title,
            document_type=document_type,
            legal_source=legal_source or self.default_legal_source,
            ministries=self._extract_list(field_map.get("ministry")),
            subunits=self._extract_list(field_map.get("subunit")),
            legal_areas=self._extract_legal_areas(field_map.get("legalArea")),
            applies_to=self._extract_list(field_map.get("appliesTo")),
            authority_refs=self._extract_links(field_map.get("basedOn")),
            date_in_force=date_in_force,
            date_of_publication=date_of_publication,
            last_updated=last_updated,
            last_change_in_force=last_change_in_force,
            last_changed_by=self._extract_text(field_map.get("lastChangedBy")),
            misc_information=None,
        )

    def _parse_sections(self, main: Tag, fallback_ref_id: str) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        for order, node in enumerate(main.select("article.legalArticle"), start=1):
            section_id = node.get("id") or node.get("data-lovdata-URL") or f"{fallback_ref_id}::section-{order}"
            ref_id = node.get("data-lovdata-URL") or fallback_ref_id
            heading_el = node.select_one(".legalArticleHeader")
            heading = heading_el.get_text(" ", strip=True) if heading_el else None
            text = self._clean_text(node.get_text(" ", strip=True))
            html = str(node)
            parent = self._find_parent_section_id(node)
            level = self._estimate_level(node)

            sections.append(
                DocumentSection(
                    section_id=section_id,
                    ref_id=ref_id,
                    heading=heading,
                    text=text,
                    html=html,
                    level=level,
                    order=order,
                    parent_section_id=parent,
                )
            )
        return sections

    def _parse_relationships(
        self, header: Tag, main: Optional[Tag], ref_id: Optional[str]
    ) -> List[DocumentRelationship]:
        relationships: List[DocumentRelationship] = []
        field_map = self._collect_dt_dd_pairs(header)

        for target in self._extract_doc_links(field_map.get("basedOn")):
            relationships.append(DocumentRelationship(relation_type="BASED_ON", target_ref_id=target))

        for target in self._extract_doc_links(field_map.get("changesToDocuments")):
            relationships.append(DocumentRelationship(relation_type="CHANGES", target_ref_id=target))

        for target in self._extract_doc_links(field_map.get("lastChangedBy")):
            relationships.append(DocumentRelationship(relation_type="CHANGES", target_ref_id=target))

        for target in self._extract_doc_links(field_map.get("repeals")):
            relationships.append(DocumentRelationship(relation_type="REPEALS", target_ref_id=target))

        if main is not None:
            for node in main.select(".changesToParent"):
                for target in self._extract_doc_links(node):
                    relationships.append(DocumentRelationship(relation_type="CHANGES", target_ref_id=target))

            for target in self._extract_doc_links(main):
                relationships.append(DocumentRelationship(relation_type="RELATED", target_ref_id=target))

        deduped: List[DocumentRelationship] = []
        seen: set[tuple[str, str]] = set()
        for rel in relationships:
            if ref_id and rel.target_ref_id == ref_id:
                continue
            key = (rel.relation_type, rel.target_ref_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rel)
        return deduped

    def _extract_misc_information(self, header: Tag) -> Optional[str]:
        field_map = self._collect_dt_dd_pairs(header)
        if "miscInformation" not in field_map:
            return None
        dd = field_map["miscInformation"]
        text = dd.get_text(" ", strip=True)
        return text or None

    @staticmethod
    def _collect_dt_dd_pairs(header: Tag) -> dict[str, Tag]:
        lookup: dict[str, Tag] = {}
        for dt in header.find_all("dt"):
            classes = dt.get("class", [])
            if not classes:
                continue
            key = classes[0]
            dd = dt.find_next_sibling("dd")
            if dd:
                lookup[key] = dd
        return lookup

    @staticmethod
    def _extract_text(node: Optional[Tag]) -> Optional[str]:
        if node is None:
            return None
        text = node.get_text(" ", strip=True)
        return text or None

    @staticmethod
    def _extract_list(node: Optional[Tag]) -> List[str]:
        if node is None:
            return []
        items = [item.get_text(" ", strip=True) for item in node.find_all("li")]
        if items:
            return [item for item in items if item]
        text = node.get_text(" ", strip=True)
        return [text] if text else []

    @staticmethod
    def _extract_links(node: Optional[Tag]) -> List[str]:
        if node is None:
            return []
        links = [a.get("href") or a.get_text(" ", strip=True) for a in node.find_all("a")]
        return [link for link in links if link]

    def _extract_doc_links(self, node: Optional[Tag]) -> List[str]:
        if node is None:
            return []
        normalized: List[str] = []
        for link in self._extract_links(node):
            ref_id = self._normalize_ref_id(link)
            if ref_id is None:
                continue
            normalized.append(ref_id)
        return normalized

    @staticmethod
    def _normalize_ref_id(link: str) -> Optional[str]:
        text = (link or "").strip()
        if not text:
            return None
        clean = text.split("#", 1)[0].split("?", 1)[0]
        if clean.startswith("http://") or clean.startswith("https://"):
            marker = "lovdata.no/"
            if marker in clean:
                clean = clean.split(marker, 1)[1]
            else:
                return None
        if clean.startswith("/"):
            clean = clean[1:]
        if clean.startswith("SF/"):
            clean = clean[3:]
        if clean.startswith("NL/"):
            clean = clean[3:]
        if clean.startswith("lov/") or clean.startswith("forskrift/"):
            parts = clean.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
        return None

    @staticmethod
    def _extract_legal_areas(node: Optional[Tag]) -> List[str]:
        if node is None:
            return []
        areas: List[str] = []
        for anchor in node.find_all("a"):
            if anchor.has_attr("title"):
                areas.append(anchor["title"])
            else:
                areas.append(anchor.get_text(" ", strip=True))
        return areas

    @staticmethod
    def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        cleaned = value.split(",")[0].strip()
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _find_parent_section_id(node: Tag) -> Optional[str]:
        ancestor = node.find_parent("article", class_="legalArticle")
        if ancestor is None:
            return None
        if ancestor is node:
            return None
        return ancestor.get("id")

    @staticmethod
    def _estimate_level(node: Tag) -> int:
        level = 0
        parent = node.find_parent("article", class_="legalArticle")
        while parent:
            level += 1
            parent = parent.find_parent("article", class_="legalArticle")
        return level

    @staticmethod
    def _infer_types(ref_id: Optional[str], dok_id: Optional[str]) -> Tuple[str, Optional[str]]:
        if dok_id and "/" in dok_id:
            legal_source = dok_id.split("/", 1)[0]
        else:
            legal_source = None

        document_type = "unknown"
        if ref_id:
            if ref_id.startswith("lov/"):
                document_type = "law"
            elif ref_id.startswith("forskrift/"):
                document_type = "forskrift"

        return document_type, legal_source


if __name__ == "__main__":
    extractor = LovdataExtractor()
    document = extractor.parse_file(Path("gjeldende-lover/nl/nl-18140517-000.xml"))
    print(document)
