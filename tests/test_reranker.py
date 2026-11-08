from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evidence_rag.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def reranker() -> CrossEncoderReranker:
    """Create a reranker with a mocked cross-encoder model."""
    with patch("evidence_rag.retrieval.reranker.CrossEncoder") as mock_cross_encoder:
        mock_cross_encoder.return_value = MagicMock()
        instance = CrossEncoderReranker()

    return instance


def test_reranker_orders_passages_by_score(
    reranker: CrossEncoderReranker,
) -> None:
    passages = [
        "Glaucoma affects the optic nerve.",
        "Deep learning detects diabetic retinopathy in fundus images.",
        "Cataracts cause clouding of the lens.",
    ]

    reranker.model.predict.return_value = np.array([0.1, 0.9, 0.3])

    results = reranker.rerank(
        "How does deep learning detect diabetic retinopathy?",
        passages,
        top_k=3,
    )

    assert [result.index for result in results] == [1, 2, 0]
    assert results[0].text == passages[1]
    assert results[0].score == pytest.approx(0.9)


def test_reranker_respects_top_k(
    reranker: CrossEncoderReranker,
) -> None:
    reranker.model.predict.return_value = np.array([0.2, 0.8, 0.5])

    results = reranker.rerank(
        "Research question",
        ["Passage A", "Passage B", "Passage C"],
        top_k=2,
    )

    assert len(results) == 2
    assert [result.index for result in results] == [1, 2]


def test_reranker_handles_empty_passages(
    reranker: CrossEncoderReranker,
) -> None:
    results = reranker.rerank("Research question", [])

    assert results == []
    reranker.model.predict.assert_not_called()


def test_reranker_rejects_empty_query(
    reranker: CrossEncoderReranker,
) -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        reranker.rerank("   ", ["Valid passage"])


def test_reranker_rejects_invalid_top_k(
    reranker: CrossEncoderReranker,
) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        reranker.rerank("Research question", ["Valid passage"], top_k=0)


def test_reranker_rejects_empty_passage(
    reranker: CrossEncoderReranker,
) -> None:
    with pytest.raises(ValueError, match="Passages must not be empty"):
        reranker.rerank("Research question", ["Valid passage", "  "])


def test_reranker_rejects_mismatched_scores(
    reranker: CrossEncoderReranker,
) -> None:
    reranker.model.predict.return_value = np.array([0.5])

    with pytest.raises(
        ValueError,
        match="Model returned an unexpected number of scores",
    ):
        reranker.rerank(
            "Research question",
            ["Passage A", "Passage B"],
        )


def test_reranker_preserves_order_for_tied_scores(
    reranker: CrossEncoderReranker,
) -> None:
    reranker.model.predict.return_value = np.array([0.7, 0.7, 0.2])

    results = reranker.rerank(
        "Research question",
        ["Passage A", "Passage B", "Passage C"],
    )

    assert [result.index for result in results] == [0, 1, 2]
