"""Norwegian (Bokmål) prompts for the user document retrieval pipeline."""
from __future__ import annotations

from agents.models import UserDocQueryPayload, UserDocQuerySet

# Status messages
STATUS_READING_DOCS = "Reading documents..."
STATUS_EVALUATING_DOCS = "Evaluating documents..."
STATUS_EVALUATING_QUERY = "Evaluating documents against request..."
STATUS_REVIEWING_DOCS = "Reviewing documents..."
ERROR_USER_DOC_FAILED = "Could not retrieve from documents now. Continuing without document context."

# --- Query generation prompts ---

USER_DOC_QUERY_INSTRUCTIONS = """Du skal generere 2-3 ULIKE søkepar mot brukerens opplastede dokumenter (kontrakter, avtaler, brev).

FORMÅL: Finn KONTEKST i brukerens konkrete dokumenter som hjelper med å besvare spørsmålet.
Dette handler IKKE om å finne juridiske prinsipper — det handler om å finne KLAUSULER og FAKTA i brukerens dokumenter.

Hvert søkepar skal dekke en ULIK del av dokumentet eller en ulik vinkling:
- Tenk på hvilke klausuler/avsnitt som er relevante (oppsigelse, varighet, pris, ansvar, mislighold)
- Hvilke termer pleier å stå i denne typen dokument?
- Hvilke fakta fra brukerens situasjon er viktige?

Hvert søkepar har:
1. semantic_query: En naturlig norsk frase som beskriver hva vi skal finne (10–20 ord)
2. lexical_query: Kontraktstermer og nøkkelord som faktisk står i dokumenter (5–10 norske termer)

Regler for semantic_query:
- Fokuser på dokumentinnhold: «Hva sier [dokumenttype] om [tema/klausul]?»

Regler for lexical_query:
- Bruk termer som faktisk står I kontrakter/avtaler
- Bruk kun ord fra samtalen + standard kontraktsvokabular
- INGEN meta-termer som «søk», «finn», «dokument»

MÅ IKKE:
- Introdusere parter/datoer/beløp som ikke er nevnt i samtalen
- Bruke lovnavn eller §-henvisninger (det er for juridisk søk)
- Generere generelle spørsmål — vær spesifikk om klausuler
"""

USER_DOC_RETRIEVE_IDENTITY = "Du genererer søk for å finne relevante klausuler i brukerens dokumenter."


def build_user_doc_query_examples() -> str:
    """Bygg eksempler for ulike dokumenttyper."""
    examples = [
        (
            "Bruker har lastet opp arbeidsavtale. Spør: 'Kan arbeidsgiver si meg opp når jeg er syk?'",
            UserDocQuerySet(queries=[
                UserDocQueryPayload(
                    semantic_query="Hva sier arbeidsavtalen om oppsigelse, prøvetid og oppsigelsesfrist?",
                    lexical_query="arbeidsavtale oppsigelse oppsigelsesfrist prøvetid varighet arbeidsgiver",
                ),
                UserDocQueryPayload(
                    semantic_query="Hva sier avtalen om sykefravær, sykdom og fravær fra arbeid?",
                    lexical_query="sykefravær sykdom fravær sykemelding arbeidstaker rettigheter",
                ),
            ]),
        ),
        (
            "Bruker har lastet opp husleiekontrakt. Spør: 'Kan utleier kaste meg ut?'",
            UserDocQuerySet(queries=[
                UserDocQueryPayload(
                    semantic_query="Hvilke vilkår gjelder for oppsigelse og heving i leiekontrakten?",
                    lexical_query="leiekontrakt oppsigelse heving mislighold varsel frist utleier leietaker",
                ),
                UserDocQueryPayload(
                    semantic_query="Hva sier kontrakten om leietakers plikter og brudd på avtalen?",
                    lexical_query="leietaker plikter vedlikehold brudd avtale husordensregler",
                ),
            ]),
        ),
        (
            "Bruker har lastet opp kjøpsavtale. Spør: 'Hva skjer hvis produktet er ødelagt?'",
            UserDocQuerySet(queries=[
                UserDocQueryPayload(
                    semantic_query="Hva sier kjøpsavtalen om reklamasjon, garanti og mangler?",
                    lexical_query="kjøpsavtale reklamasjon garanti mangel ansvar erstatning retur",
                ),
                UserDocQueryPayload(
                    semantic_query="Hva sier avtalen om levering, risiko og godkjenning av vare?",
                    lexical_query="levering risiko overgang godkjenning mottak skade transport",
                ),
            ]),
        ),
    ]

    lines = ["EKSEMPLER:", ""]
    for context, output in examples:
        lines.append(context)
        lines.append(f"→ {output.model_dump_json()}")
        lines.append("")

    return "\n".join(lines)


def build_user_doc_query_prompt() -> str:
    """Bygg systemprompten for søkgenerering mot brukerens dokumenter."""
    return "\n".join([USER_DOC_RETRIEVE_IDENTITY, "", USER_DOC_QUERY_INSTRUCTIONS, "", build_user_doc_query_examples()])


SUMMARY_INSTRUCTIONS = """Oppsummer dokumentinnholdet basert på de hentede utdragene.
- 2-3 setninger som beskriver dokumenttypen og hovedinnholdet
- Nevn viktige klausuler/temaer som er funnet (f.eks. oppsigelse, varighet, ansvar)
- Skriv på norsk
- Vær konkret om hva dokumentet handler om"""


def build_summary_input(filename: str, chunks: list) -> str:
    """Bygg input-prompt for oppsummering av dokumentutdrag."""
    chunk_texts = "\n\n".join(
        f"[Utdrag {i+1}]: {chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    return f"Dokumentnavn: {filename}\n\nUtdrag fra dokumentet:\n{chunk_texts}"
