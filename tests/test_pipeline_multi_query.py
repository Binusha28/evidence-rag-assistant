"""Integration tests for multi-query evidence retrieval."""

from unittest.mock import patch

import pytest

from evidence_rag.generation.rag_pipeline import (
    RAGConfig,
    ResearchRAGPipeline,
)
from evidence_rag.ingestion.pipeline import DocumentChunk


def make_chunk(
    chunk_id: str,
    source: str,
    page_number: int,
    text: str,
) -> DocumentChunk:
    """Create a valid document chunk for pipeline tests."""

    return DocumentChunk(
        chunk_id=chunk_id,
        source=source,
        page_number=page_number,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_multi_query_preserves_source_metadata():
    pipeline = ResearchRAGPipeline(
        llm_client=None,
        config=RAGConfig(top_k=2, candidate_k=3),
    )

    chunks = [
        make_chunk(
            "cnn:p3:c0",
            "cnn.pdf",
            3,
            "CNN retinal lesion detection uses feature maps.",
        ),
        make_chunk(
            "vit:p3:c0",
            "vit.pdf",
            3,
            "Vision Transformers process image patches.",
        ),
    ]

    with patch(
        "evidence_rag.generation.rag_pipeline.ingest_pdf",
        return_value=chunks,
    ):
        pipeline.load_document("test.pdf")

    evidence = pipeline.retrieve_multi_query_evidence(
        [
            "CNN retinal lesion feature maps",
            "Vision Transformer image patches",
        ]
    )

    assert len(evidence) == 2
    assert {item.source for item in evidence} == {
        "cnn.pdf",
        "vit.pdf",
    }
    assert {item.chunk_id for item in evidence} == {
        "cnn:p3:c0",
        "vit:p3:c0",
    }
    assert {item.citation for item in evidence} == {
        "[E1]",
        "[E2]",
    }


def test_multi_query_requires_loaded_documents():
    pipeline = ResearchRAGPipeline(llm_client=None)

    with pytest.raises(RuntimeError):
        pipeline.retrieve_multi_query_evidence(["CNN", "ViT"])


@pytest.mark.parametrize(
    "queries",
    [
        [],
        [""],
        ["CNN", " "],
    ],
)
def test_multi_query_rejects_invalid_queries(queries):
    pipeline = ResearchRAGPipeline(llm_client=None)

    chunks = [
        make_chunk(
            "cnn:p1:c0",
            "cnn.pdf",
            1,
            "CNN retinal lesion detection uses feature maps.",
        )
    ]

    with patch(
        "evidence_rag.generation.rag_pipeline.ingest_pdf",
        return_value=chunks,
    ):
        pipeline.load_document("test.pdf")

    with pytest.raises(ValueError):
        pipeline.retrieve_multi_query_evidence(queries)
