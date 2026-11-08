"""Extract citation-linked claims from generated answers."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CitedClaim:
    text: str
    citation_ids: tuple[str, ...]


_ABBREVIATIONS = (
    "et al.",
    "Dr.",
    "Prof.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "e.g.",
    "i.e.",
)

_CITATION_PATTERN = re.compile(r"\[E\d+\]")

# Conservative splitting for independently cited comparison clauses.
_COMPARISON_CONNECTOR = re.compile(
    r"^\s*,?\s*(?:while|whereas)\s+",
    flags=re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split sentences while preserving common abbreviations."""
    if not text.strip():
        return []

    protected = text.strip()

    for abbreviation in _ABBREVIATIONS:
        pattern = re.compile(
            re.escape(abbreviation),
            flags=re.IGNORECASE,
        )
        protected = pattern.sub(
            lambda match: match.group().replace(".", "<DOT>"),
            protected,
        )

    sentences = re.split(
        r"(?<=[.!?])(?:\s+|(?=[A-Z]))",
        protected,
    )

    return [sentence.replace("<DOT>", ".").strip() for sentence in sentences if sentence.strip()]


def _split_independently_cited_clauses(
    sentence: str,
) -> list[str]:
    """Split a narrow, explicitly cited comparison pattern.

    Only split when a citation immediately precedes a comparison
    connector. Other constructions remain intact for verification.
    """
    matches = list(_CITATION_PATTERN.finditer(sentence))

    if len(matches) < 2:
        return [sentence]

    boundaries: list[int] = []

    for match in matches[:-1]:
        remainder = sentence[match.end() :]

        connector = _COMPARISON_CONNECTOR.match(remainder)

        if connector is None:
            continue

        following = remainder[connector.end() :]

        if not _CITATION_PATTERN.search(following):
            continue

        boundaries.append(match.end())

    if not boundaries:
        return [sentence]

    clauses: list[str] = []
    start = 0

    for boundary in boundaries:
        clause = sentence[start:boundary].strip()

        if clause:
            clauses.append(clause)

        start = boundary

    final_clause = sentence[start:].strip()

    if final_clause:
        clauses.append(final_clause)

    cleaned: list[str] = []

    for clause in clauses:
        clause = _COMPARISON_CONNECTOR.sub("", clause)
        clause = clause.strip(" ,")

        if clause:
            cleaned.append(clause)

    return cleaned or [sentence]


def extract_cited_claims(answer: str) -> list[CitedClaim]:
    """Extract independently cited claims where safely possible."""
    claims: list[CitedClaim] = []

    for sentence in split_sentences(answer):
        clauses = _split_independently_cited_clauses(sentence)

        for clause in clauses:
            citations = tuple(dict.fromkeys(_CITATION_PATTERN.findall(clause)))

            if not citations:
                continue

            claim_text = _CITATION_PATTERN.sub(
                "",
                clause,
            ).strip()

            # Remove spaces introduced before punctuation
            # when citation identifiers are removed.
            claim_text = re.sub(
                r"\s+([.,!?;:])",
                r"\1",
                claim_text,
            )

            if not claim_text:
                continue

            claims.append(
                CitedClaim(
                    text=claim_text,
                    citation_ids=citations,
                )
            )

    return claims


def _has_uncited_trailing_clause(sentence: str) -> bool:
    """Detect an uncited factual clause following a citation.

    This is a conservative syntactic check rather than
    a complete semantic clause parser.
    """
    citations = list(_CITATION_PATTERN.finditer(sentence))

    if not citations:
        return False

    trailing = sentence[citations[-1].end() :]

    return bool(
        re.match(
            r"^\s*,\s*(?:and|but|while|whereas)\s+\S+",
            trailing,
            flags=re.IGNORECASE,
        )
    )


def extract_uncited_sentences(answer: str) -> list[str]:
    """Identify uncited sentences and suspicious trailing clauses."""
    uncited: list[str] = []

    for sentence in split_sentences(answer):
        has_citation = bool(_CITATION_PATTERN.search(sentence))

        if not has_citation:
            uncited.append(sentence)
            continue

        if _has_uncited_trailing_clause(sentence):
            uncited.append(sentence)

    return uncited
