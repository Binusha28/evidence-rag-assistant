"""Test actionable feedback for failed claim verification."""

from types import SimpleNamespace

from evidence_rag.verification.verification_feedback import (
    build_verification_feedback,
)


def make_report(
    assessments=(),
    uncited_sentences=(),
    leading_citations=(),
):
    return SimpleNamespace(
        assessments=assessments,
        uncited_sentences=uncited_sentences,
        leading_citations=leading_citations,
    )


def make_assessment(claim, citation, label):
    return SimpleNamespace(
        claim=claim,
        citation=citation,
        label=label,
    )


def test_neutral_feedback_identifies_exact_claim():
    report = make_report(
        assessments=(
            make_assessment(
                "These regions are classified.",
                "[E1]",
                "neutral",
            ),
        ),
    )

    feedback = build_verification_feedback(report)

    assert "These regions are classified." in feedback
    assert "[E1]" in feedback
    assert "self-contained" in feedback
    assert "omit it" in feedback


def test_contradiction_feedback_identifies_claim():
    report = make_report(
        assessments=(
            make_assessment(
                "The model achieved 100% accuracy.",
                "[E2]",
                "contradiction",
            ),
        ),
    )

    feedback = build_verification_feedback(report)

    assert "The model achieved 100% accuracy." in feedback
    assert "[E2]" in feedback
    assert "correct it" in feedback


def test_uncited_feedback_identifies_sentence():
    report = make_report(
        uncited_sentences=("The model always performs perfectly.",),
    )

    feedback = build_verification_feedback(report)

    assert "The model always performs perfectly." in feedback
    assert "supporting citation" in feedback


def test_missing_evidence_feedback():
    report = make_report(
        assessments=(
            make_assessment(
                "A new finding.",
                "[E9]",
                "missing_evidence",
            ),
        ),
    )

    feedback = build_verification_feedback(report)

    assert "[E9]" in feedback
    assert "available citation identifiers" in feedback


def test_empty_report_returns_generic_feedback():
    report = make_report()

    feedback = build_verification_feedback(report)

    assert "Review the answer" in feedback
