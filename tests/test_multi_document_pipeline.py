"""Tests for indexing and retrieving evidence from multiple PDFs."""

from unittest.mock import patch

import pytest

from evidence_rag.generation.rag_pipeline import ResearchRAGPipeline
from evidence_rag.ingestion.pipeline import DocumentChunk


def make_chunk(source: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{source}:p1:c0",
        source=source,
        page_number=1,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_load_documents_combines_sources():
    documents = {
        "paper_a.pdf": [
            make_chunk(
                "paper_a.pdf",
                "Neural networks learn representations from data.",
            )
        ],
        "paper_b.pdf": [
            make_chunk(
                "paper_b.pdf",
                "Transformer models use attention mechanisms.",
            )
        ],
    }

    pipeline = ResearchRAGPipeline(llm_client=None)

    with (
        patch(
            "evidence_rag.generation.rag_pipeline.ingest_pdf",
            side_effect=lambda path, **kwargs: documents[str(path)],
        ),
        patch(
            "evidence_rag.generation.rag_pipeline.filter_reference_chunks",
            side_effect=lambda chunks: chunks,
        ),
        patch(
            "evidence_rag.generation.rag_pipeline.filter_low_information_chunks",
            side_effect=lambda chunks: chunks,
        ),
    ):
        count = pipeline.load_documents(["paper_a.pdf", "paper_b.pdf"])

    assert count == 2
    assert len(pipeline.chunks) == 2
    assert {chunk.source for chunk in pipeline.chunks} == {
        "paper_a.pdf",
        "paper_b.pdf",
    }

    evidence = pipeline.retrieve_evidence("What do transformer models use?")

    assert evidence
    assert all(item.source in {"paper_a.pdf", "paper_b.pdf"} for item in evidence)
    assert all(item.page_number == 1 for item in evidence)


def test_load_documents_rejects_empty_input():
    pipeline = ResearchRAGPipeline(llm_client=None)

    with pytest.raises(ValueError, match="At least one PDF"):
        pipeline.load_documents([])


def test_load_documents_rejects_duplicate_chunk_ids():
    duplicate = make_chunk(
        "paper_a.pdf",
        "Neural networks learn representations from data.",
    )

    pipeline = ResearchRAGPipeline(llm_client=None)

    with (
        patch(
            "evidence_rag.generation.rag_pipeline.ingest_pdf",
            return_value=[duplicate],
        ),
        patch(
            "evidence_rag.generation.rag_pipeline.filter_reference_chunks",
            side_effect=lambda chunks: chunks,
        ),
        patch(
            "evidence_rag.generation.rag_pipeline.filter_low_information_chunks",
            side_effect=lambda chunks: chunks,
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Duplicate chunk identifier",
        ):
            pipeline.load_documents(["paper_a.pdf", "paper_a.pdf"])
