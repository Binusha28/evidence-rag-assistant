"""Conservatively recover independently verified answer sentences."""

import re

from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
)
from evidence_rag.verification.claim_extractor import (
    extract_cited_claims,
    split_sentences,
)

_CITATION = re.compile(r"\[E\d+\]")

_END_CITATION = re.compile(r"\s+\[E\d+\][.!?]$")


def recover_verified_sentences(
    answer: str,
    report: AnswerVerificationReport,
) -> str | None:
    """Recover only complete sentences with independently supported citations.

    This function does not make the final verification decision.
    Its output must be verified again before presentation.
    """
    if report.leading_citations:
        return None

    supported = {
        (item.claim, item.citation) for item in report.assessments if item.label == "entailment"
    }

    rejected = {
        (item.claim, item.citation) for item in report.assessments if item.label != "entailment"
    }

    recovered: list[str] = []

    for sentence in split_sentences(answer):
        citations = _CITATION.findall(sentence)

        if len(citations) != 1:
            continue

        if not _END_CITATION.search(sentence):
            continue

        claims = extract_cited_claims(sentence)

        if len(claims) != 1:
            continue

        claim = claims[0]

        if claim.citation_ids != (citations[0],):
            continue

        key = (claim.text, citations[0])

        if key in rejected or key not in supported:
            continue

        if sentence in report.uncited_sentences:
            continue

        recovered.append(sentence)

    if not recovered:
        return None

    return " ".join(recovered)
