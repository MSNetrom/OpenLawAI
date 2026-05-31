"""Norwegian (Bokmål) prompts and status strings shared across modes."""
from __future__ import annotations

from typing import Any, Dict, List

from agents.models import ModeName


SYSTEM_IDENTITY = """Du er en juridisk assistent for det norske lovverket.

DIN VERDI: Du gir svar med kildehenvisninger til konkrete lovparagrafer og forskrifter. Det er dette som skiller deg fra generell AI-chat — du finner og siterer faktiske rettskilder.

Hentede juridiske kilder er ditt primære grunnlag. Du supplerer med din egen juridiske kunnskap der det er naturlig, uten å flagge at noe mangler.

Brukeren kan laste opp dokumenter som en del av henvendelsen, men dette er ikke juridiske kilder, dette er bare dokumenter som er en del av henvendelsen fra brukeren.

Lover og forskrifter er kilder som du henter, og er det primære grunnlaget for å besvare henvendelsen fra brukeren."""


MODE_GUIDANCE: Dict[ModeName, str] = {
    "clarify": "spørr brukeren om manglende informasjon, som forhindrer deg fra å gi et godt og presist svar",
    "retrieve": "hent relevante juridiske kilder før svar",
    "user_doc_retrieve": "søk i brukerens opplastede dokumenter; bør brukes for å hente relevant informasjon fra opplastede dokumenter",
    "generate": "generer et dokument (PDF/DOCX) for nedlasting",
    "answer": "svar til brukeren med kilder basert på tilgjengelig kontekst",
}

SET_MODE_DESCRIPTION_PREFIX = "Velg neste modus i assistentens handlingspipeline."


def doc_titles_for_context(documents: List[Dict[str, Any]]) -> str:
    """Compact document titles for context."""
    if not documents:
        return "Ingen juridiske kilder hentet."
    lines = [f"- {d['title']} ({d['work_ref_id']}, {d['document_type']})" for d in documents]
    return f"Juridiske kilder ({len(documents)}):\n" + "\n".join(lines)


# Status messages emitted during processing
STATUS_ANALYZING = "Analyserer henvendelse..."
STATUS_SEARCHING = "Søker i juridiske kilder..."
STATUS_ANSWERING = "Formulerer svar..."
STATUS_CLARIFYING = "Formulerer oppfølgingsspørsmål..."
STATUS_GENERATING = "Genererer dokument..."
STATUS_PROCESSING_DOCS = "Behandler opplastede dokumenter..."
STATUS_SEARCHING_USER_DOCS = "Søker i opplastede dokumenter..."

# Infrastructure prompts used by chat_manager
JSON_REPAIR_SUFFIX = "\n\nKRITISK: Returner kun gyldig JSON som matcher skjemaet nøyaktig."
SUMMARY_PREVIOUS_LABEL = "FORRIGE OPPSUMMERING:"
SUMMARY_NEW_MESSAGES_LABEL = "NYE MELDINGER:"


def summary_instructions(max_words: int) -> str:
    return (
        "Oppdater oppsummeringen med de nye meldingene. "
        "Bruk stikkord og korte setninger — ikke prosa. "
        "Behold: brukerens mål, viktige fakta, tall, paragrafer, og hva som er besvart. "
        f"Maks {max_words} ord. Kun oppsummering."
    )
