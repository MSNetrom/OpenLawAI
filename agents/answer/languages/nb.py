"""Norwegian prompts and builder functions for AnswerMode."""
from __future__ import annotations

import json
from typing import Dict, List

from agents.shared import SYSTEM_IDENTITY


ANSWER_INSTRUCTIONS = (
    "Gi et strukturert svar basert på kildene. Start rett på innholdet uten innledende etikett eller overskrift.\n"
    "- Svar direkte uten oppfølgingsspørsmål; hvis mer detalj er nyttig, avslutt med en kort invitasjon.\n"
    "- Første setning skal gå rett på sak — aldri start med etiketter som 'Kort svar:', 'Kort vurdering:', 'Oppsummering:' e.l.\n"
    "\n"
    "FORMATERING:\n"
    "- Bruk **fet skrift** for viktige begreper, lovnavn og nøkkelord.\n"
    "- Bruk ### overskrifter for å dele opp lengre svar i logiske seksjoner.\n"
    "- Bruk punktlister der det er naturlig for oversikt.\n"
    "\n"
    "BRUK AV EGEN KUNNSKAP:\n"
    "- Du KAN og SKAL supplere hentede kilder med din egen juridiske kunnskap.\n"
    "- Nevn relevante lover naturlig uten å flagge at de ikke ble hentet.\n"
    "- Si ALDRI 'Jeg fant ikke', 'Jeg har ikke tilgang til', eller tilby å søke etter flere dokumenter.\n"
    "- Når du nevner en lov fra egen kunnskap (ikke i kildene), referer til den ved navn uten lenke.\n"
    "- Kun bruk [(Lovnavn § X)](lovdata_url) formatet for paragrafer fra HENTEDE kilder.\n"
    "- Bruk ALLE hentede kilder som er relevante — ikke bare de mest åpenbare. Hvis en hentet lov er relevant, skal den refereres.\n"
    "- Dekk alle relevante rettsområder fra egen kunnskap, ikke bare det som er hentet.\n"
    "\n"
    "KILDEHENVISNINGER:\n"
    "- Format: [(Lovnavn § paragraf)](https://lovdata.no/{lovdata_url})\n"
    "- Eksempel: [(Arbeidsmiljøloven § 15-11)](https://lovdata.no/NL/lov/2005-06-17-62/§15-11)\n"
    "- lovdata_url finner du i hvert chunk under 'lovdata_url'-feltet.\n"
    "- Referer KUN til paragrafer som finnes i de vedlagte kildene med lenkeformat. Ikke oppgi paragrafnummer du ikke finner i dokumentene.\n"
    "- Avslutt med en 'Kilder:' seksjon. Utelat 'Kilder:'-seksjonen kun for rene hilsener og småprat.\n"
    "- Det er SVÆRT VIKTIG at du gir kildehenvisninger, på formatet spesifisert i eksempelet, både for kilder som er i teksten, og for kildene i 'Kilder:' seksjonen.\n"
    "\n"
    "KILDESPORING:\n"
    "- I 'used_source_ids' SKAL du oppgi work_ref_id for alle juridiske dokumenter du faktisk refererer til i svaret.\n"
    "- work_ref_id finner du i hvert dokument under 'work_ref_id'-feltet.\n"
)

USER_DOC_INSTRUCTIONS = """
NÅR BRUKEREN HAR LASTET OPP DOKUMENTER:
- Analyser dokumentene grundig punkt for punkt
- Kommenter konkrete formuleringer og vilkår i dokumentet
- Vurder om vilkårene er i tråd med gjeldende lov
- Påpek eventuelle mangler eller problematiske klausuler"""

GENERATED_DOC_INSTRUCTIONS = """
GENERERT DOKUMENT TILGJENGELIG:
- Informer brukeren om at dokumentet er generert og klart til nedlasting.
- Inkluder nedlastingslenken som markdown-lenke: [filnavn](url) - kopier lenken nøyaktig fra listen over.
- IKKE skriv URL-en som ren tekst, bruk ALLTID markdown-lenkeformat."""

STATUS_FORMULATING = "Formulerer svar..."


def build_static_instructions(
    has_user_docs: bool = False,
    has_unannounced_docs: bool = False,
) -> str:
    """Build the static system instructions for AnswerMode (cacheable by OpenAI).

    Dynamic content (documents, chunks) goes into input_items, not here.
    """
    parts = [SYSTEM_IDENTITY, "", ANSWER_INSTRUCTIONS]

    if has_user_docs:
        parts.append(USER_DOC_INSTRUCTIONS)

    if has_unannounced_docs:
        parts.append(GENERATED_DOC_INSTRUCTIONS)

    return "\n".join(parts)


def build_document_context_message(
    documents: List[Dict],
    user_doc_results: List[Dict] | None = None,
    all_generated_docs: List | None = None,
) -> dict:
    """Build a single context message containing all document content.

    This goes into input_items as a prefix before conversation messages,
    enabling prefix caching on the static instructions.
    """
    from agents.models import settings

    parts: List[str] = []

    parts.append(f"JURIDISKE DOKUMENTER:\n{json.dumps(documents, ensure_ascii=False)}")

    if user_doc_results:
        trimmed_user_docs = user_doc_results[:settings.max_user_docs]
        parts.append(f"BRUKERENS OPPLASTEDE DOKUMENTER:\n{json.dumps(trimmed_user_docs, ensure_ascii=False)}")

    if all_generated_docs:
        doc_links = [f"[{doc.filename}](/api/documents/{doc.id}/)" for doc in all_generated_docs]
        parts.append("GENERERTE DOKUMENTER (kopier disse lenkene nøyaktig):\n" + "\n".join(doc_links))

    return {"role": "user", "content": "\n\n".join(parts)}
