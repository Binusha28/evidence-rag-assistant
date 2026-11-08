"""Tests for conservative semantic citation repair."""

from types import SimpleNamespace

from evidence_rag.generation.evidence import Evidence
from evidence_rag.verification.citation_repair import SemanticCitationRepair


class FakeVerifier:
    def verify(self, claim, evidence):
        if claim == "A CNN generates a feature map.":
            return SimpleNamespace(
                label="entailment",
                confidence=0.99,
            )

        return SimpleNamespace(
            label="neutral",
            confidence=0.99,
        )


def make_evidence():
    return [
        Evidence(
            evidence_id="E1",
            chunk_id="chunk-1",
            source="research.pdf",
            page_number=3,
            text="A CNN generates a feature map.",
            retrieval_score=1.0,
        )
    ]


def test_repairs_supported_uncited_sentence():
    repairer = SemanticCitationRepair(verifier=FakeVerifier())

    result = repairer.repair(
        answer="A CNN generates a feature map.",
        evidence=make_evidence(),
    )

    assert result.repaired_answer == "A CNN generates a feature map [E1]."
    assert result.changed


def test_does_not_repair_unsupported_sentence():
    repairer = SemanticCitationRepair(verifier=FakeVerifier())

    result = repairer.repair(
        answer="The model achieves 100% accuracy.",
        evidence=make_evidence(),
    )

    assert result.repaired_answer == "The model achieves 100% accuracy."
    assert not result.changed
    assert result.unrepaired_sentences == ("The model achieves 100% accuracy.",)


def test_preserves_existing_citation():
    repairer = SemanticCitationRepair(verifier=FakeVerifier())

    result = repairer.repair(
        answer="A CNN generates a feature map [E1].",
        evidence=make_evidence(),
    )

    assert result.repaired_answer == "A CNN generates a feature map [E1]."
    assert not result.changed


def test_does_not_replace_invalid_citation():
    repairer = SemanticCitationRepair(verifier=FakeVerifier())

    result = repairer.repair(
        answer="A CNN generates a feature map [E9].",
        evidence=make_evidence(),
    )

    assert result.repaired_answer == "A CNN generates a feature map [E9]."
    assert not result.changed


def test_rejects_invalid_confidence_threshold():
    try:
        SemanticCitationRepair(
            verifier=FakeVerifier(),
            min_confidence=1.5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_does_not_hide_uncited_sentence_after_citation():
    repairer = SemanticCitationRepair(verifier=FakeVerifier())

    answer = "A CNN generates a feature map [E1].The model achieves 100% accuracy."

    result = repairer.repair(
        answer=answer,
        evidence=make_evidence(),
    )

    assert "The model achieves 100% accuracy." in result.unrepaired_sentences
    assert "[E1]. The model achieves 100% accuracy." in result.repaired_answer


def test_preserves_mixed_valid_and_invalid_citations():
    repairer = SemanticCitationRepair(verifier=FakeVerifier())

    answer = "A CNN generates a feature map [E1] [E99]."

    result = repairer.repair(
        answer=answer,
        evidence=make_evidence(),
    )

    assert result.repaired_answer == answer
    assert not result.changed
