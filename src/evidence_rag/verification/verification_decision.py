"""Make conservative decisions from answer verification reports."""

from dataclasses import dataclass

from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
)


@dataclass(frozen=True)
class VerificationDecision:
    status: str
    reasons: tuple[str, ...]
    safe_to_present_as_verified: bool


def decide_verification(
    report: AnswerVerificationReport,
) -> VerificationDecision:
    """Classify an answer using the available verification checks."""
    reasons = []

    if not report.assessments:
        reasons.append("No cited claims were assessed.")

    if report.uncited_sentences:
        reasons.append(f"{len(report.uncited_sentences)} uncited sentence(s) detected.")

    if report.leading_citations:
        reasons.append("Incorrect citation placement detected.")

    labels = {item.label for item in report.assessments}

    if "missing_evidence" in labels:
        reasons.append("One or more citations reference missing evidence.")

    if "contradiction" in labels:
        reasons.append("One or more claims contradict cited evidence.")

    if "neutral" in labels:
        reasons.append("One or more claims lack sufficient evidence support.")

    if any(
        label not in {"entailment", "neutral", "contradiction", "missing_evidence"}
        for label in labels
    ):
        reasons.append("Unknown verification result detected.")

    if report.is_verified:
        return VerificationDecision(
            status="passed_checks",
            reasons=(),
            safe_to_present_as_verified=True,
        )

    return VerificationDecision(
        status="needs_review",
        reasons=tuple(reasons) or ("Verification checks did not pass.",),
        safe_to_present_as_verified=False,
    )
