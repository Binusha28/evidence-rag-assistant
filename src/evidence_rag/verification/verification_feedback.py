"""Build actionable feedback for answers that fail verification."""

from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
)


def build_verification_feedback(
    report: AnswerVerificationReport,
) -> str:
    """Convert verification failures into specific correction instructions."""
    instructions: list[str] = []

    if report.uncited_sentences:
        instructions.append(
            "Every factual sentence must include its supporting "
            "citation before the final punctuation. "
            "Do not leave factual sentences uncited."
        )

        for sentence in report.uncited_sentences:
            instructions.append(
                f"Uncited sentence: {sentence!r}. Add a supporting citation or remove the sentence."
            )

    if report.leading_citations:
        instructions.append(
            "Never start a sentence with a citation identifier. "
            "Place citations after the factual claims they support."
        )

    # Detect compound claims assessed against multiple citations.
    citations_by_claim: dict[str, set[str]] = {}

    for assessment in report.assessments:
        citations_by_claim.setdefault(assessment.claim, set()).add(assessment.citation)

    unsupported_claims = {
        assessment.claim for assessment in report.assessments if assessment.label != "entailment"
    }

    compound_claims = [
        claim
        for claim, citations in citations_by_claim.items()
        if len(citations) > 1 and claim in unsupported_claims
    ]

    if compound_claims:
        instructions.append(
            "The previous answer contains compound statements with "
            "multiple citations that do not independently support "
            "the complete statement. Rewrite these as separate, "
            "self-contained factual sentences. Each sentence must "
            "express one independently supported claim and end "
            "with its own supporting citation. Do not merely remove "
            "citations or split text mechanically. Preserve the "
            "scientific meaning and omit unsupported claims."
        )

    for assessment in report.assessments:
        if assessment.label == "neutral":
            instructions.append(
                f"Claim requiring revision: {assessment.claim!r}. "
                f"Citation: {assessment.citation}. "
                "The cited evidence did not establish this claim "
                "according to the semantic verifier. "
                "Rewrite the claim using precise terminology from "
                "the cited passage. "
                "Make the sentence self-contained and avoid ambiguous "
                "references such as 'these regions', 'this method', "
                "'they', or 'it'. "
                "Do not introduce additional scientific details. "
                "If the evidence does not support the claim, omit it."
            )

        elif assessment.label == "contradiction":
            instructions.append(
                f"Contradictory claim: {assessment.claim!r}. "
                f"Citation: {assessment.citation}. "
                "Remove the claim or correct it using only "
                "information explicitly stated in the cited evidence."
            )

        elif assessment.label == "missing_evidence":
            instructions.append(
                f"Claim with missing evidence: {assessment.claim!r}. "
                f"Citation: {assessment.citation}. "
                "Use only available citation identifiers. "
                "If no supporting evidence exists, omit the claim."
            )

    if not instructions:
        return "Review the answer for evidence support and citation accuracy."

    return "\n".join(
        f"{number}. {instruction}" for number, instruction in enumerate(instructions, start=1)
    )
