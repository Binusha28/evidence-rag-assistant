"""Tests for evidence-grounded answer generation."""

from unittest.mock import MagicMock

import pytest

from evidence_rag.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE,
    EvidenceAnswerGenerator,
)
from evidence_rag.generation.evidence import Evidence


def make_evidence() -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id="research.pdf:p4:c2",
        source="research.pdf",
        page_number=4,
        text="CNNs analyse retinal fundus photographs.",
        retrieval_score=0.85,
    )


def test_generate_grounded_answer():
    client = MagicMock()
    client.generate.return_value = "CNNs analyse retinal fundus photographs [E1]."

    generator = EvidenceAnswerGenerator(client)
    result = generator.generate(
        "How do CNNs analyse retinal images?",
        [make_evidence()],
    )

    assert "[E1]" in result.answer
    assert result.evidence[0].page_number == 4

    kwargs = client.generate.call_args.kwargs
    assert "research.pdf" in kwargs["prompt"]
    assert "Page: 4" in kwargs["prompt"]
    assert "CNNs analyse retinal fundus photographs." in kwargs["prompt"]


def test_empty_evidence():
    client = MagicMock()
    generator = EvidenceAnswerGenerator(client)

    result = generator.generate("What is the diagnosis?", [])

    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.evidence == ()
    client.generate.assert_not_called()


def test_empty_question():
    client = MagicMock()
    generator = EvidenceAnswerGenerator(client)

    with pytest.raises(ValueError, match="Question must not be empty"):
        generator.generate(" ", [make_evidence()])


def test_evidence_is_preserved():
    client = MagicMock()
    client.generate.return_value = "Supported answer [E1]."

    evidence = make_evidence()
    generator = EvidenceAnswerGenerator(client)

    result = generator.generate("Explain the evidence.", [evidence])

    assert result.question == "Explain the evidence."
    assert result.evidence == (evidence,)


def test_rejects_unknown_citation():
    client = MagicMock()
    client.generate.return_value = "Unsupported citation [E99]."

    generator = EvidenceAnswerGenerator(client)

    with pytest.raises(ValueError, match="unknown citations"):
        generator.generate(
            "What does the research show?",
            [make_evidence()],
        )


def test_rejects_answer_without_citations():
    client = MagicMock()
    client.generate.return_value = "An answer without references."

    generator = EvidenceAnswerGenerator(client)

    with pytest.raises(ValueError, match="does not contain any"):
        generator.generate(
            "What does the research show?",
            [make_evidence()],
        )
