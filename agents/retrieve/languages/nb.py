"""Norwegian (Bokmål) prompts for the retrieval pipeline."""
from __future__ import annotations

from agents.models import (
    ChatHistory,
    RetrievalQueryPayload,
    RetrievalQuerySet,
    RetrievalRefinePayload,
)

# Status messages
STATUS_SEARCHING = "Searching for documents..."
ERROR_RETRIEVAL_FAILED = "Could not retrieve legal sources right now. Continuing without sources."
QUERIES_SUMMARY_PREVIOUS_ROUND = "(Sources retrieved in previous retrieval round in the same conversation.)"


def occupancy_status(pool_count: int, max_docs: int) -> str:
    return f"Documents in pool: {pool_count} (target: max {max_docs})"


def user_doc_context(chat_history: ChatHistory) -> list[dict]:
    """Build context messages with user doc summaries for the retrieve agent."""
    docs = chat_history.metadata.user_docs.documents
    summaries = [
        f"- {d.filename}: {d.summary}"
        for d in docs
        if d.status == "ready" and d.summary
    ]
    if not summaries:
        return []
    return [{"role": "user", "content": "BRUKERENS DOKUMENTER (oppsummering):\n" + "\n".join(summaries)}]


# --- Query Generation Prompts ---

QUERY_INSTRUCTIONS = """Formuler søk for å finne relevante juridiske kilder i en norsk juridisk database med hybrid-søk.

Du skal generere opptil 7 søkepar med to typer: "conceptual" og "targeted".

QUERY-TYPER:
1. conceptual (standard): Brede temasøk som dekker ulike juridiske fasetter.
   - semantic_query: Naturlig norsk setning som beskriver det juridiske konseptet (10-20 ord). IKKE nevn spesifikke lovnavn.
   - lexical_query: Nøkkelord som finnes i juridiske dokumenter (5-10 termer). Kan inkludere lovnavn.
   - query_type: "conceptual"

2. targeted (ved behov): Søk etter en spesifikk lov eller forskrift ved navn.
   - Bruk når brukeren nevner en bestemt lov, eller du vet at en sentral lov er relevant.
   - semantic_query: Beskriv hva brukeren leter etter i den loven.
   - lexical_query: Lovens/forskriftens EKSAKTE NAVN pluss relevante nøkkelord.
   - query_type: "targeted"

Bruk conceptual som standard. Bruk targeted kun når du vet at en spesifikk lov er relevant.
Ikke begrens deg til en fast fordeling — la spørsmålet avgjøre miksen.

GENERELLE REGLER:
- Generer søk om det KONKRETE juridiske temaet brukeren spør om.
- Hver query skal dekke en ULIK vinkling/fasett — ikke bare omformuler samme spørsmål.
- Tenk på: Hvilke ulike rettsområder og juridiske temaer er relevante? Maksimer BREDDE.
- IKKE generer meta-spørsmål om juridisk metode, rettskildelære, eller hvordan man gjør juridisk research.
- IKKE be om presisering eller still oppfølgingsspørsmål til brukeren, du lager søkefraser for å søke i juridiske kilder.
- Ikke bruk web-søkeoperatorer som site:, url:, eller andre søkeoperatorer.
- Ingen meta-termer som 'søk', 'finn', 'kilder', 'rettskilder'.
- Hvis brukeren har lastet opp dokumenter (kontrakter, avtaler), dekomponér dokumentets temaer til relevante juridiske søk."""


def build_query_examples() -> str:
    """Build query examples showing context → multi-query transformation."""
    examples = [
        (
            "Bruker: 'Kan jeg bli sagt opp når jeg er syk?'",
            RetrievalQuerySet(queries=[
                RetrievalQueryPayload(
                    semantic_query="Hvilke regler gjelder for oppsigelse av arbeidstaker under sykefravær?",
                    lexical_query="oppsigelse sykefravær oppsigelsesvern saklig grunn sykmeldt arbeidstaker",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Hva er arbeidstakers rett til sykepenger og ytelser ved langvarig sykdom?",
                    lexical_query="sykepenger folketrygdloven arbeidsgiverperiode sykdom rettigheter arbeidstaker",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Regler for oppsigelsesvern i arbeidsmiljøloven ved sykdom",
                    lexical_query="arbeidsmiljøloven oppsigelse sykdom vern",
                    query_type="targeted",
                ),
                RetrievalQueryPayload(
                    semantic_query="Arbeidsgivers tilretteleggingsplikt for syk arbeidstaker og krav til oppfølging",
                    lexical_query="tilrettelegging arbeidsgiver plikt sykefravær oppfølging dialogmøte",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Regler om sykepenger i folketrygdloven og arbeidsgiverperiode",
                    lexical_query="folketrygdloven sykepenger arbeidsgiverperiode",
                    query_type="targeted",
                ),
            ]),
        ),
        (
            "Bruker: 'Hva sier personopplysningsloven om behandling av sensitive opplysninger?'",
            RetrievalQuerySet(queries=[
                RetrievalQueryPayload(
                    semantic_query="Regler for behandling av sensitive personopplysninger og særlige kategorier av data",
                    lexical_query="sensitive personopplysninger behandlingsgrunnlag samtykke helseopplysninger",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Bestemmelser om sensitive opplysninger i personopplysningsloven",
                    lexical_query="personopplysningsloven sensitive opplysninger behandling",
                    query_type="targeted",
                ),
                RetrievalQueryPayload(
                    semantic_query="Krav til sikkerhet og dokumentasjon ved behandling av personopplysninger",
                    lexical_query="behandlingsansvarlig databehandler sikkerhetstiltak tekniske organisatoriske",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Personvernforordningen GDPR artikkel 9 om særlige kategorier personopplysninger",
                    lexical_query="personvernforordningen GDPR artikkel 9 særlige kategorier",
                    query_type="targeted",
                ),
            ]),
        ),
        (
            "Bruker: 'Kan utleier kaste meg ut fordi jeg har hund?'",
            RetrievalQuerySet(queries=[
                RetrievalQueryPayload(
                    semantic_query="Hva er vilkårene for oppsigelse og utkastelse av leietaker?",
                    lexical_query="oppsigelse utkastelse heving leietaker utleier vilkår husleie",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Kan utleier forby dyrehold i leiekontrakt og hva gjelder rettslig?",
                    lexical_query="dyrehold husdyr leiekontrakt forbud leietaker husleie",
                    query_type="conceptual",
                ),
                RetrievalQueryPayload(
                    semantic_query="Bestemmelser i husleieloven om oppsigelse og heving av leieforhold",
                    lexical_query="husleieloven oppsigelse heving mislighold",
                    query_type="targeted",
                ),
                RetrievalQueryPayload(
                    semantic_query="Regler for fravikelse og tvangsfullbyrdelse ved leieforhold",
                    lexical_query="fravikelse tvangsfullbyrdelse namsmann leietaker utleier",
                    query_type="conceptual",
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


RETRIEVE_IDENTITY = "Du genererer søk for en norsk juridisk database."


def build_query_prompt() -> str:
    """Build the system prompt for query generation."""
    parts = [RETRIEVE_IDENTITY, "", QUERY_INSTRUCTIONS, build_query_examples()]
    return "\n".join(parts)


# --- Refine Decision Prompts ---

REFINE_INSTRUCTIONS = """Du skal vurdere dokumentene som er hentet så langt.

OPPGAVE:
1. Vurder om noen dokumenter er IRRELEVANTE for brukerens spørsmål. Legg dem i drop_work_ref_ids.
2. Vurder om det MANGLER viktige kilder. Tenk: hvilke sentrale lover/forskrifter trengs for et komplett svar?
3. Hvis noe mangler, kan du generere nye søk i new_queries.

VIKTIG — DROPP FRIGJØR PLASSER:
- Å droppe et dokument frigjør plassen det brukte.
- Hvis poolen er full men du ser irrelevante dokumenter, DROPP dem og bruk de frigjorte plassene til å hente viktigere kilder.
- Tenk strategisk: bytt ut svake/irrelevante dokumenter med mer sentrale kilder.

KAPASITET:
- Du har plass til maks {max_docs} dokumenter totalt, uavhengig av dokumenttype.
- Behold kun de mest relevante. Dokumenttype (lov/forskrift) spiller ingen rolle for kapasiteten.

ALT ER VALGFRITT — gjør kun det som trengs:
- Du trenger IKKE droppe dokumenter med mindre de genuint er irrelevante.
- Du trenger IKKE generere nye queries — selv om du dropper dokumenter, kan resten være tilstrekkelig.
- Du KAN generere queries uten å droppe noe (hvis det er ledige plasser og noe mangler).
- Du kan generere 0-7 queries, i hvilken som helst miks av typer.

QUERY-TYPER I REFINE:
- query_type="targeted": Foretrekk dette i refine. Du har sett kildene og vet hvilken spesifikk lov som mangler.
  Legg lovens eksakte navn i lexical_query. Eksempel: lexical_query="ferieloven ferie rettigheter".
- query_type="conceptual": Bruk for bredere tematiske hull du ikke kan knytte til en spesifikk lov.

REGLER:
- Antall nye queries <= ledige plasser + antall dokumenter du dropper.
- Nye queries skal søke etter MANGLENDE kilder — ikke kilder du allerede har.
- Hver work_ref_id kan kun stå én gang i drop_work_ref_ids.
- Ikke bruk web-søkeoperatorer som site:, url:, eller andre søkeoperatorer.

COVERAGE_SUMMARY:
- Skriv én kort setning som beskriver hvilke juridiske temaer de gjenværende dokumentene samlet sett dekker.
- Fokuser på temaer, ikke titler. Ingen detaljer om droppede dokumenter."""


def build_refine_examples() -> str:
    """Build examples for refinement decisions."""
    examples = [
        {
            "scenario": "Fornøyd med kildene, ingen endringer",
            "output": RetrievalRefinePayload(
                new_queries=[],
                drop_work_ref_ids=[],
                coverage_summary="Arbeidsrettslige regler om oppsigelse, stillingsvern og arbeidsmiljø.",
            ),
        },
        {
            "scenario": "Dropper irrelevante dokumenter, men resten er tilstrekkelig — ingen nye queries",
            "output": RetrievalRefinePayload(
                new_queries=[],
                drop_work_ref_ids=["forskrift/2003-12-12-1566", "lov/1992-06-26-86"],
                coverage_summary="Regler om husleieforhold, depositum og utleiers vedlikeholdsplikt.",
            ),
        },
        {
            "scenario": "Pool er full. Dropper irrelevant dokument og bruker den frigjorte plassen til å hente ferieloven",
            "output": RetrievalRefinePayload(
                new_queries=[
                    RetrievalQueryPayload(
                        semantic_query="Rett til ferie og feriepenger for arbeidstakere",
                        lexical_query="ferieloven ferie feriepenger rettigheter",
                        query_type="targeted",
                    ),
                ],
                drop_work_ref_ids=["forskrift/1999-12-22-1379"],
                coverage_summary="Regler om arbeidsforhold, oppsigelsesfrister og ferie.",
            ),
        },
    ]

    lines = ["EKSEMPLER:", ""]
    for ex in examples:
        lines.append(f"Situasjon: {ex['scenario']}")
        lines.append(f"→ {ex['output'].model_dump_json()}")
        lines.append("")

    return "\n".join(lines)


FINAL_REFINE_INSTRUCTIONS = """Du skal gjøre en SISTE vurdering av dokumentene. Det er ingen flere søkerunder etter dette.

OPPGAVE:
1. Vurder om noen dokumenter er IRRELEVANTE for brukerens spørsmål. Legg dem i drop_work_ref_ids.
2. Dropp kun dokumenter som genuint ikke er relevante — ikke dropp for å droppe.
3. Behold maks {max_docs} dokumenter totalt (uavhengig av type).

REGLER:
- Sett new_queries til tom liste []. Det er ingen flere søkerunder.
- Kun dropp dokumenter som er genuint irrelevante.
- Hver work_ref_id kan kun stå én gang i drop_work_ref_ids.

COVERAGE_SUMMARY:
- Skriv én kort setning som beskriver hvilke juridiske temaer de gjenværende dokumentene samlet sett dekker.
- Fokuser på temaer, ikke titler. Ingen detaljer om droppede dokumenter."""


def build_refine_prompt(
    *,
    max_docs: int,
    round_index: int,
    max_refines: int,
    occupancy_str: str,
    queries_summary: str,
    pool_trimmed_json: str,
) -> str:
    """Build the full refine system prompt for a given round."""
    return f"""{REFINE_INSTRUCTIONS.format(max_docs=max_docs)}

{build_refine_examples()}

Dette er runde {round_index + 1} av maks {max_refines}.

PLASS I DOKUMENTPOOL:
{occupancy_str}

Queries som ble brukt for å hente disse dokumentene:
{queries_summary}

Dokumenter (full innhold, trimmet):
{pool_trimmed_json}"""


def build_final_refine_prompt(*, max_docs: int, pool_trimmed_json: str) -> str:
    """Build the final refine system prompt."""
    return f"""{FINAL_REFINE_INSTRUCTIONS.format(max_docs=max_docs)}

Dokumenter (full innhold, trimmet):
{pool_trimmed_json}"""

