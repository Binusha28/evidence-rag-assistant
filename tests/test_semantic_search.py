import numpy as np
import pytest

from evidence_rag.retrieval.semantic_search import SemanticRetriever


class FakeEmbeddingModel:
    """Provide predictable embeddings for retrieval tests."""

    def __init__(self) -> None:
        self.vectors = {
            "medical research": [1.0, 0.0],
            "healthcare evidence": [0.8, 0.6],
            "weather forecast": [0.0, 1.0],
            "medical question": [1.0, 0.0],
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [self.vectors[text] for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def retriever() -> SemanticRetriever:
    model = FakeEmbeddingModel()
    return SemanticRetriever(model)  # type: ignore[arg-type]


def test_search_ranks_relevant_passages_first(
    retriever: SemanticRetriever,
) -> None:
    retriever.index(
        [
            "weather forecast",
            "healthcare evidence",
            "medical research",
        ]
    )

    results = retriever.search("medical question", top_k=3)

    assert [result.text for result in results] == [
        "medical research",
        "healthcare evidence",
        "weather forecast",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].index == 2


def test_top_k_limits_results(retriever: SemanticRetriever) -> None:
    retriever.index(["medical research", "weather forecast"])

    results = retriever.search("medical question", top_k=1)

    assert len(results) == 1
    assert results[0].text == "medical research"


def test_search_before_index_raises_error(
    retriever: SemanticRetriever,
) -> None:
    with pytest.raises(RuntimeError, match="Index passages"):
        retriever.search("medical question")


def test_empty_index_raises_error(retriever: SemanticRetriever) -> None:
    with pytest.raises(ValueError, match="At least one passage"):
        retriever.index([])


def test_empty_query_raises_error(retriever: SemanticRetriever) -> None:
    retriever.index(["medical research"])

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("   ")


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(
    retriever: SemanticRetriever,
    top_k: int,
) -> None:
    retriever.index(["medical research"])

    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("medical question", top_k=top_k)


def test_top_k_larger_than_index(retriever: SemanticRetriever) -> None:
    retriever.index(["medical research", "weather forecast"])

    results = retriever.search("medical question", top_k=10)

    assert len(results) == 2
