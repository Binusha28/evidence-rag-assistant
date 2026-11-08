from collections.abc import Sequence

import numpy as np
import pytest

from evidence_rag.ingestion.pipeline import DocumentChunk
from evidence_rag.retrieval.document_search import DocumentSearcher
from evidence_rag.retrieval.semantic_search import SemanticRetriever


class FakeEmbeddingModel:
    """Return predictable embeddings without downloading a model."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = {
            "weather forecast": [0.0, 1.0],
            "retinal disease detection": [1.0, 0.0],
            "medical image analysis": [0.8, 0.6],
            "detect eye diseases": [1.0, 0.0],
        }
        return np.asarray(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def searcher() -> DocumentSearcher:
    model = FakeEmbeddingModel()
    retriever = SemanticRetriever(model)  # type: ignore[arg-type]
    return DocumentSearcher(retriever)


@pytest.fixture
def document_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="weather.pdf:p1:c0",
            source="weather.pdf",
            page_number=1,
            text="weather forecast",
            start_word=0,
            end_word=2,
        ),
        DocumentChunk(
            chunk_id="research.pdf:p4:c0",
            source="research.pdf",
            page_number=4,
            text="retinal disease detection",
            start_word=10,
            end_word=13,
        ),
        DocumentChunk(
            chunk_id="research.pdf:p7:c0",
            source="research.pdf",
            page_number=7,
            text="medical image analysis",
            start_word=20,
            end_word=23,
        ),
    ]


def test_results_preserve_source_metadata(
    searcher: DocumentSearcher,
    document_chunks: list[DocumentChunk],
) -> None:
    searcher.index(document_chunks)

    results = searcher.search("detect eye diseases", top_k=3)

    assert len(results) == 3

    assert results[0].text == "retinal disease detection"
    assert results[0].source == "research.pdf"
    assert results[0].page_number == 4
    assert results[0].chunk_id == "research.pdf:p4:c0"
    assert results[0].start_word == 10
    assert results[0].end_word == 13
    assert results[0].score == pytest.approx(1.0)

    assert results[1].source == "research.pdf"
    assert results[1].page_number == 7

    assert results[2].source == "weather.pdf"
    assert results[2].page_number == 1


def test_top_k_preserves_correct_citation(
    searcher: DocumentSearcher,
    document_chunks: list[DocumentChunk],
) -> None:
    searcher.index(document_chunks)

    results = searcher.search("detect eye diseases", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "research.pdf:p4:c0"


def test_empty_document_index(searcher: DocumentSearcher) -> None:
    with pytest.raises(ValueError, match="At least one document chunk"):
        searcher.index([])


def test_search_before_index(searcher: DocumentSearcher) -> None:
    with pytest.raises(RuntimeError, match="Index passages"):
        searcher.search("detect eye diseases")


def test_reindex_updates_metadata(
    searcher: DocumentSearcher,
    document_chunks: list[DocumentChunk],
) -> None:
    searcher.index(document_chunks)
    searcher.index([document_chunks[1]])

    results = searcher.search("detect eye diseases", top_k=3)

    assert len(results) == 1
    assert results[0].source == "research.pdf"
    assert results[0].page_number == 4
