"""Norwegian (Bokmål) prompts for the generate pipeline."""
from __future__ import annotations

import json
from typing import Dict, List

from agents.models import settings
from agents.shared import SYSTEM_IDENTITY


# --- Prompt constants ---

GENERATE_INSTRUCTIONS = """Du skal generere et strukturert dokument basert på brukerens forespørsel.

VIKTIG: Produser et komplett, velstrukturert dokument i GitHub-flavored Markdown.
- Bruk overskrifter (#, ##, ###) for å strukturere dokumentet
- Bruk punktlister og nummererte lister der det passer
- Inkluder tabeller om relevant
- Henvis til konkrete lovparagrafer og forskrifter inline i teksten
- IKKE pakk markdown i kodefences

KRITISK — DOKUMENTFORMAT:
Markdown-feltet skal inneholde KUN selve dokumentet.
- INGEN oppfølgingsspørsmål, forslag, eller "vil du at jeg skal…"
- INGEN "Praktiske merknader", "Anbefalinger", eller rådgivende seksjoner
- INGEN chatmeldinger, tilbud om videre hjelp, eller konversasjonselementer
- Signaturblokk/signaturfelt KUN hvis dokumenttypen krever det (f.eks. kontrakter, avtaler, fullmakter) — ALDRI på informasjonsdokumenter, oversikter, eller veiledninger
- INGEN «klikk her»-instruksjoner, URL-er, eller lenker til Lovdata
- INGEN separat kildeseksjon på slutten — kildehenvisningene skal stå inline i teksten
- Dokumentet avsluttes etter siste innholdsseksjon, uten vedlegg eller appendiks"""

GENERATE_STATIC_INSTRUCTIONS = SYSTEM_IDENTITY + "\n\n" + GENERATE_INSTRUCTIONS

# Status / error messages
STATUS_GENERATING = "Generating document..."
ERROR_GENERATE_FAILED = "Could not generate the document. Please try again."


def build_document_context_message(documents: List[Dict], user_doc_results: List[Dict] | None = None) -> dict:
    """Build document context message for input_items prefix."""
    parts: List[str] = [f"JURIDISKE KILDER:\n{json.dumps(documents, ensure_ascii=False)}"]

    if user_doc_results:
        parts.append(f"BRUKERENS DOKUMENTER:\n{json.dumps(user_doc_results[:settings.max_user_docs], ensure_ascii=False)}")

    return {"role": "user", "content": "\n\n".join(parts)}
