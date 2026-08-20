"""Norwegian (Bokmål) prompts for the clarify pipeline."""
from __future__ import annotations

from typing import Dict, List

from agents.shared import SYSTEM_IDENTITY, _doc_titles_for_context

# Status messages
STATUS_PREPARING = "Preparing follow-up questions..."

CLARIFY_INSTRUCTIONS = """Still oppfølgingsspørsmål for å forstå brukerens behov bedre, og for å kunne gi et godt og presist svar.

Spørsmålene skal være direkte knyttet til temaet og lovområdet brukeren allerede har nevnt. Bruk hele samtalehistorikken som grunnlag, ikke bare siste melding.
Ikke spør om informasjon som allerede er gitt i samtalen.
Hvis et lovnavn eller juridisk område er nevnt tidligere, bruk det som tema og ikke spør hvilket lovområde dette gjelder.
Hvis brukeren ber om «generell oversikt» eller «generelt svar», tolkes det som en oversikt over det allerede nevnte temaet.
Minst ett spørsmål skal eksplisitt referere til det nevnte lovnavnet eller området når det finnes.
Ikke spør etter «originalt spørsmål».
Hvis brukeren allerede har oppgitt kontekst eller perspektiv, bruk det og spør heller om konkrete forhold som gjør svaret mer presist.
Unngå generiske metaspørsmål som ikke avgrenser saken.
Ikke gi svar eller forklaring nå; kun oppfølgingsspørsmål.

Formater output som en full markdown-melding med:
1. En kort overskrift "### Oppklaringer"
2. 2-4 korte, konkrete oppfølgingsspørsmål som punktliste
3. Til slutt én kort setning som inviterer til et generelt svar hvis brukeren ikke vil svare

KILDEHENVISNINGER:
- Når du refererer til en kilde i spørsmålene, bruk markdown-lenker.
- Format: [(Lovnavn § paragraf)](https://lovdata.no/{lovdata_url})
- Eksempel: 'Antall ansatte påvirker krav til verneombud [(Arbeidsmiljøloven § 6-1)](url)'

KILDESPORING:
- I 'used_source_ids' SKAL du oppgi work_ref_id for alle juridiske dokumenter du faktisk refererer til i spørsmålene.
- work_ref_id finner du i hvert dokument under 'work_ref_id'-feltet.
"""


def build_system_prompt(documents: List[Dict], user_doc_summaries: List[str]) -> str:
    """Build the system prompt for ClarifyMode with compact context.

    Uses document titles (not full chunks) and user doc summaries only.
    """
    parts = [SYSTEM_IDENTITY, "", CLARIFY_INSTRUCTIONS, "", _doc_titles_for_context(documents)]

    if user_doc_summaries:
        parts.append(f"\nBrukerdokumenter ({len(user_doc_summaries)}):")
        parts.extend(f"- {s}" for s in user_doc_summaries)

    return "\n".join(parts)
