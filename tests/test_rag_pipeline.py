"""Tests for the end-to-end research RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)
from evidence_rag.ingestion.pipeline import DocumentChunk


def make_chunk(
    chunk_id: str,
    text: str,
    page_number: int = 1,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source="paper.pdf",
        page_number=page_number,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


@pytest.mark.parametrize(
    "config",
    [
        {"top_k": 0},
        {"chunk_size": 0},
        {"overlap": -1},
        {"chunk_size": 10, "overlap": 10},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        RAGConfig(**config)


def test_ask_before_loading():
    pipeline = ResearchRAGPipeline(MagicMock())

    with pytest.raises(RuntimeError, match="Load a PDF"):
        pipeline.ask("What is diabetic retinopathy?")


@patch("evidence_rag.generation.rag_pipeline.ingest_pdf")
def test_load_document(mock_ingest):
    mock_ingest.return_value = [
        make_chunk(
            "paper.pdf:p1:c0",
            "CNNs analyse retinal fundus photographs.",
        ),
        make_chunk(
            "paper.pdf:p2:c0",
            "Diabetic retinopathy affects retinal blood vessels.",
            page_number=2,
        ),
    ]

    pipeline = ResearchRAGPipeline(MagicMock())
    count = pipeline.load_document("paper.pdf")

    assert count == 2
    assert len(pipeline.chunks) == 2


@patch("evidence_rag.generation.rag_pipeline.ingest_pdf")
def test_empty_document(mock_ingest):
    mock_ingest.return_value = []

    pipeline = ResearchRAGPipeline(MagicMock())

    with pytest.raises(ValueError, match="No usable passages"):
        pipeline.load_document("paper.pdf")


@patch("evidence_rag.generation.rag_pipeline.ingest_pdf")
def test_generate_answer_from_retrieved_evidence(mock_ingest):
    mock_ingest.return_value = [
        make_chunk(
            "paper.pdf:p4:c2",
            "CNNs detect diabetic retinopathy using retinal photographs.",
            page_number=4,
        ),
        make_chunk(
            "paper.pdf:p5:c0",
            "Machine learning supports retinal image analysis.",
            page_number=5,
        ),
    ]

    client = MagicMock()
    client.generate.return_value = "CNNs analyse retinal photographs [E1]."

    pipeline = ResearchRAGPipeline(
        client,
        config=RAGConfig(top_k=2),
    )

    pipeline.load_document("paper.pdf")
    result = pipeline.ask("How do CNNs detect diabetic retinopathy?")

    assert "[E1]" in result.answer
    assert len(result.evidence) == 2
    assert all(item.source == "paper.pdf" for item in result.evidence)

    client.generate.assert_called_once()


@patch("evidence_rag.generation.rag_pipeline.ingest_pdf")
def test_empty_question_after_loading(mock_ingest):
    mock_ingest.return_value = [make_chunk("paper.pdf:p1:c0", "Research passage.")]

    pipeline = ResearchRAGPipeline(MagicMock())
    pipeline.load_document("paper.pdf")

    with pytest.raises(ValueError, match="Question must not be empty"):
        pipeline.ask(" ")
