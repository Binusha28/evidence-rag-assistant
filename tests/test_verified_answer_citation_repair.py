"""Test citation repair integration with verified answer generation."""

from types import SimpleNamespace

from evidence_rag.generation.evidence import Evidence
from evidence_rag.generation.verified_answer import VerifiedAnswerGenerator
from evidence_rag.verification.citation_repair import SemanticCitationRepair


class FakeAnswerGenerator:
    def generate(self, question, evidence, feedback=None):
        return SimpleNamespace(
            answer="A CNN generates a feature map.",
            evidence=tuple(evidence),
        )


class FakeVerifier:
    def verify(self, claim, evidence):
        if "A CNN generates a feature map" in claim:
            return SimpleNamespace(
                label="entailment",
                confidence=0.99,
            )

        return SimpleNamespace(
            label="neutral",
            confidence=0.99,
        )


def test_verified_generator_repairs_supported_uncited_sentence():
    evidence = [
        Evidence(
            evidence_id="E1",
            chunk_id="chunk-1",
            source="research.pdf",
            page_number=3,
            text="A CNN generates a feature map.",
            retrieval_score=1.0,
        )
    ]

    verifier = FakeVerifier()

    generator = VerifiedAnswerGenerator(
        answer_generator=FakeAnswerGenerator(),
        verifier=verifier,
        citation_repair=SemanticCitationRepair(
            verifier=verifier,
            min_confidence=0.95,
        ),
    )

    result = generator.generate(
        question="What does the CNN generate?",
        evidence=evidence,
    )

    assert result.answer == "A CNN generates a feature map [E1]."
    assert result.is_accepted
    assert result.verification.uncited_sentences == ()
    assert len(result.verification.assessments) == 1
    assert result.verification.assessments[0].label == "entailment"
