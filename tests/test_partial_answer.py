"""Tests for conservative recovery of verified answer sentences."""

from evidence_rag.generation.partial_answer import (
    recover_verified_sentences,
)
from evidence_rag.verification.answer_verification import (
    AnswerVerificationReport,
)
from evidence_rag.verification.claim_verification_pipeline import (
    ClaimEvidenceAssessment,
)


def assessment(claim, citation, label):
    return ClaimEvidenceAssessment(
        claim=claim,
        citation=citation,
        label=label,
        confidence=0.95,
    )


def report(*assessments, uncited=(), leading=()):
    return AnswerVerificationReport(
        assessments=tuple(assessments),
        uncited_sentences=tuple(uncited),
        leading_citations=tuple(leading),
    )


def test_recovers_supported_sentence_only():
    answer = "CNNs analyse retinal images [E1]. AI guarantees perfect diagnosis [E2]."

    verification = report(
        assessment(
            "CNNs analyse retinal images.",
            "[E1]",
            "entailment",
        ),
        assessment(
            "AI guarantees perfect diagnosis.",
            "[E2]",
            "neutral",
        ),
    )

    assert recover_verified_sentences(answer, verification) == ("CNNs analyse retinal images [E1].")


def test_rejects_unsupported_sentence():
    answer = "AI guarantees perfect diagnosis [E1]."

    verification = report(
        assessment(
            "AI guarantees perfect diagnosis.",
            "[E1]",
            "neutral",
        )
    )

    assert recover_verified_sentences(answer, verification) is None


def test_rejects_compound_multi_citation_sentence():
    answer = "AI detects lesions [E1] and predicts outcomes [E2]."

    verification = report(
        assessment(
            "AI detects lesions and predicts outcomes.",
            "[E1]",
            "entailment",
        ),
        assessment(
            "AI detects lesions and predicts outcomes.",
            "[E2]",
            "neutral",
        ),
    )

    assert recover_verified_sentences(answer, verification) is None


def test_rejects_citation_not_at_sentence_end():
    answer = "AI detects lesions [E1] and improves treatment."

    verification = report(
        assessment(
            "AI detects lesions and improves treatment.",
            "[E1]",
            "entailment",
        )
    )

    assert recover_verified_sentences(answer, verification) is None


def test_rejects_leading_citation():
    answer = "[E1] CNNs analyse retinal images."

    verification = report(
        assessment(
            "CNNs analyse retinal images.",
            "[E1]",
            "entailment",
        ),
        leading=("[E1]",),
    )

    assert recover_verified_sentences(answer, verification) is None


def test_rejects_uncited_sentence():
    answer = "CNNs analyse retinal images."

    assert (
        recover_verified_sentences(
            answer,
            report(),
        )
        is None
    )


def test_preserves_fully_supported_sentences():
    answer = "CNNs analyse retinal images [E1]. Deep learning supports screening [E2]."

    verification = report(
        assessment(
            "CNNs analyse retinal images.",
            "[E1]",
            "entailment",
        ),
        assessment(
            "Deep learning supports screening.",
            "[E2]",
            "entailment",
        ),
    )

    assert recover_verified_sentences(answer, verification) == answer
