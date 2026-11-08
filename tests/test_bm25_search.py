import pytest

from evidence_rag.retrieval.bm25_search import BM25Retriever, tokenize


@pytest.fixture
def retriever() -> BM25Retriever:
    model = BM25Retriever()
    model.index(
        [
            "retinal diabetic retinopathy detection using fundus images",
            "weather forecast and rainfall prediction",
            "machine learning for medical diagnosis",
            "deep learning detects retinal disease in fundus images",
        ]
    )
    return model


def test_tokenization() -> None:
    assert tokenize("Deep Learning, AI!") == [
        "deep",
        "learning",
        "ai",
    ]


def test_search_ranks_relevant_passage_first(
    retriever: BM25Retriever,
) -> None:
    results = retriever.search(
        "diabetic retinopathy fundus",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].index == 0
    assert "diabetic retinopathy" in results[0].text


def test_top_k_limits_results(retriever: BM25Retriever) -> None:
    results = retriever.search("retinal images", top_k=2)

    assert len(results) == 2


def test_search_before_index() -> None:
    model = BM25Retriever()

    with pytest.raises(RuntimeError, match="Index passages"):
        model.search("retinal disease")


def test_empty_index() -> None:
    model = BM25Retriever()

    with pytest.raises(ValueError, match="At least one passage"):
        model.index([])


def test_empty_passage() -> None:
    model = BM25Retriever()

    with pytest.raises(ValueError, match="Passages must not be empty"):
        model.index(["valid passage", "   "])


def test_empty_query(retriever: BM25Retriever) -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("")


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(
    retriever: BM25Retriever,
    top_k: int,
) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        retriever.search("retinal disease", top_k=top_k)


def test_top_k_larger_than_index(retriever: BM25Retriever) -> None:
    results = retriever.search("retinal images", top_k=10)

    assert len(results) == 4


def test_reindex_replaces_old_passages() -> None:
    model = BM25Retriever()
    model.index(["old document about weather"])
    model.index(["new document about retinal imaging"])

    results = model.search("retinal imaging")

    assert len(results) == 1
    assert results[0].text == "new document about retinal imaging"
