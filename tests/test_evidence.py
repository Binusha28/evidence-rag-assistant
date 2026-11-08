"""Tests for structured RAG evidence and citations."""

from dataclasses import dataclass

import pytest

from evidence_rag.generation.evidence import Evidence, build_evidence


@dataclass
class FakeSearchResult:
    chunk_id: str
    source: str
    page_number: int
    text: str
    score: float


def test_evidence_citation():
    evidence = Evidence(
        evidence_id="E1",
        chunk_id="paper.pdf:p3:c2",
        source="paper.pdf",
        page_number=3,
        text="CNNs analyse retinal images.",
        retrieval_score=0.85,
    )

    assert evidence.citation == "[E1]"


def test_evidence_prompt_format():
    evidence = Evidence(
        evidence_id="E2",
        chunk_id="paper.pdf:p5:c1",
        source="paper.pdf",
        page_number=5,
        text="Sensitivity measures detection performance.",
        retrieval_score=0.72,
    )

    formatted = evidence.format_for_prompt()

    assert "[E2]" in formatted
    assert "paper.pdf" in formatted
    assert "Page: 5" in formatted
    assert "Sensitivity measures detection performance." in formatted


def test_build_evidence():
    results = [
        FakeSearchResult(
            chunk_id="paper.pdf:p1:c0",
            source="paper.pdf",
            page_number=1,
            text="First research passage.",
            score=0.9,
        ),
        FakeSearchResult(
            chunk_id="paper.pdf:p2:c0",
            source="paper.pdf",
            page_number=2,
            text="Second research passage.",
            score=0.8,
        ),
    ]

    evidence = build_evidence(results)

    assert len(evidence) == 2
    assert evidence[0].citation == "[E1]"
    assert evidence[1].citation == "[E2]"
    assert evidence[0].page_number == 1
    assert evidence[1].retrieval_score == 0.8


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("evidence_id", ""),
        ("chunk_id", ""),
        ("source", ""),
        ("page_number", 0),
        ("text", " "),
    ],
)
def test_invalid_evidence(field, invalid_value):
    values = {
        "evidence_id": "E1",
        "chunk_id": "paper.pdf:p1:c0",
        "source": "paper.pdf",
        "page_number": 1,
        "text": "Research passage.",
        "retrieval_score": 0.9,
    }

    values[field] = invalid_value

    with pytest.raises(ValueError):
        Evidence(**values)
