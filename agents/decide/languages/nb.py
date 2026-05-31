"""Norwegian (Bokmål) prompts for the decide pipeline."""
from __future__ import annotations

from typing import List

from agents.models import ModeName, UserDocsState


DECIDE_IDENTITY = "Du er en koordinator for en juridisk assistent. Velg neste steg."

STATUS_ANALYSING = "Analyserer henvendelse..."


def build_guidance(allowed_modes: List[ModeName]) -> str:
    """Build mode guidance, including only lines relevant to allowed modes."""
    lines = [
        "Velg neste steg basert på samtalehistorikk, kontekst og tone.",
        "- Hvis brukerens melding er en hilsen, småprat eller ikke inneholder noe juridisk spørsmål, bruk answer direkte. Ikke hent kilder for meldinger som 'Hei', 'Takk', 'Hva kan du hjelpe med?' eller lignende.",
        "- Skille først mellom regelspørsmål og situasjonsvurdering: Regelspørsmål ber om hva loven normalt sier; situasjonsvurdering ber om hva som gjelder i brukerens konkrete sak.",
        "- Formuleringer som 'kan jeg ...?', 'hva gjør jeg nå?' og 'er dette lovlig for meg?' er som hovedregel situasjonsvurdering.",
        "- Ja/nei-spørsmål om hva som kan skje i den konkrete saken er situasjonsvurdering.",
        "- Førsteperson alene er ikke nok for situasjonsvurdering: spørsmålet kan fortsatt være et regelspørsmål hvis brukeren ber om generelle lovregler.",
        "- Avled forventet detaljnivå fra målgruppe, tone og formulering, og bruk dette til å vurdere neste steg.",
        "- Hvis relevante juridiske kilder allerede er hentet, dekningssammendraget omfatter spørsmålet, og spørsmålet er et regelspørsmål, bruk answer. Bruk retrieve kun når temaet har endret seg tydelig eller nødvendige kilder mangler.",
        "- Når spørsmålet gjelder standardregler som normalt kan forklares generelt (for eksempel arverekkefølge, frister og beløpsgrenser), bruk answer etter retrieval.",
        "- Answer er standardvalget når det gir et presist svar uten å gjette manglende nøkkelomstendigheter.",
    ]
    if "clarify" in allowed_modes:
        lines.append("- Bruk clarify når brukerens spørsmål har mange ukjente variabler som påvirker svaret, eller når det er uklart hva det egentlige problemet eller spørsmålet er.")
        lines.append("- Etter at kilder er hentet: ved situasjonsvurdering med manglende nøkkelomstendigheter, er clarify riktig og answer feil.")
        lines.append("- Etter at kilder er hentet: hvis grunnlag, prosess eller tidslinje er uklart i en situasjonsvurdering, bruk clarify før answer.")
        lines.append("- Etter at kilder er hentet: hvis et generelt svar ville måtte dekke mange ulike situasjoner med forskjellig utfall, bruk clarify for å kunne gi et presist svar.")
        lines.append("- Foretrekk retrieve før clarify når ingen juridiske kilder er hentet ennå — kilder gir bedre grunnlag for oppfølgingsspørsmål.")
        lines.append("- Før første retrieve: hvis du ikke kan formulere minst to meningsfulle juridiske søk uten å gjette juridisk tema, bruk clarify først.")
        lines.append("- Ikke bruk clarify for presentasjonsvalg (lengde, format, detaljnivå); tilpass svaret.")
        lines.append("- Hvis oppklaringsspørsmål allerede er stilt og brukeren ikke tilfører ny informasjon, ikke spør igjen.")
        lines.append("- Én oppfølgingsrunde kan være rimelig; terskelen skal være høyere for hver nye oppklaringsrunde før svar.")
    if "user_doc_retrieve" in allowed_modes:
        lines.append("- Bruk user_doc_retrieve når brukerens dokumenter kan påvirke svaret og ikke er søkt i.")
    if "generate" in allowed_modes:
        lines.append("- Velg generate kun når brukeren eksplisitt ber om et dokument.")
    lines.append("- Foretrekk færre steg når svarkvaliteten ikke forbedres av flere steg.")
    return "\n".join(lines)


def build_status_text(
    documents: list[dict],
    user_docs: UserDocsState,
    retrieval_calls: int,
    max_retrieval_passes: int,
    retrieval_coverage: str | None,
) -> str:
    """Build status text showing current state."""
    parts: List[str] = []

    if documents:
        titles = "\n".join(f"- {doc['title']}" for doc in documents)
        header = f"Juridiske kilder ({len(documents)} dokumenter hentet)"
        if retrieval_coverage:
            header += f"\nDekning: {retrieval_coverage}"
        parts.append(f"{header}\n{titles}")
    else:
        parts.append("Du har ikke hentet noen juridiske kilder, og du må derfor hente kilder for å kunne gi kildehenvisninger.")

    if user_docs.documents:
        doc_lines = []
        for doc in user_docs.documents:
            if doc.status == "ready" and doc.summary:
                doc_lines.append(f"- {doc.filename}: {doc.summary}")
            elif doc.status == "ready" and doc.retrieved:
                doc_lines.append(f"- {doc.filename}: (hentet, ingen oppsummering)")
            elif doc.status == "ready":
                doc_lines.append(f"- {doc.filename}: (klar, ikke analysert)")
            elif doc.status == "pending":
                doc_lines.append(f"- {doc.filename}: (behandles)")
            elif doc.status == "failed":
                doc_lines.append(f"- {doc.filename}: (feilet)")
        parts.append(f"Brukerens dokumenter:\n" + "\n".join(doc_lines))
    else:
        parts.append("Til info: Brukeren har ikke lastet opp noen dokumenter.")

    parts.append(f"Retrieval: {retrieval_calls}/{max_retrieval_passes}")

    return "\n\n".join(parts)


def build_system_prompt(
    documents: list[dict],
    user_docs: UserDocsState,
    retrieval_calls: int,
    max_retrieval_passes: int,
    retrieval_coverage: str | None,
    allowed_modes: List[ModeName],
) -> str:
    """Build the full system prompt for decide agent."""
    status_text = build_status_text(
        documents=documents,
        user_docs=user_docs,
        retrieval_calls=retrieval_calls,
        max_retrieval_passes=max_retrieval_passes,
        retrieval_coverage=retrieval_coverage,
    )
    guidance = build_guidance(allowed_modes)

    return f"""{DECIDE_IDENTITY}

STATUS:
{status_text}

{guidance}
Velg én av følgende modes: {', '.join(allowed_modes)}."""
