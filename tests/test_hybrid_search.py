from dataclasses import dataclass

import pytest

from evidence_rag.retrieval.hybrid_search import HybridRetriever


@dataclass(frozen=True)
class FakeResult:
    index: int
    text: str
    score: float = 1.0


class FakeRetriever:
    def __init__(self, ranking: list[int]) -> None:
        self.ranking = ranking
        self.passages: list[str] = []

    def index(self, passages: list[str]) -> None:
        self.passages = list(passages)

    def search(self, query: str, top_k: int = 3) -> list[FakeResult]:
        return [FakeResult(index=i, text=self.passages[i]) for i in self.ranking[:top_k]]


def make_hybrid(
    semantic_ranking: list[int],
    bm25_ranking: list[int],
) -> HybridRetriever:
    semantic = FakeRetriever(semantic_ranking)
    bm25 = FakeRetriever(bm25_ranking)

    return HybridRetriever(semantic, bm25)


def test_search_before_index() -> None:
    hybrid = make_hybrid([0], [0])

    with pytest.raises(RuntimeError, match="Index passages"):
        hybrid.search("retinal disease")


def test_empty_index() -> None:
    hybrid = make_hybrid([0], [0])

    with pytest.raises(ValueError, match="At least one passage"):
        hybrid.index([])


def test_invalid_rrf_k() -> None:
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        HybridRetriever(
            FakeRetriever([0]),
            FakeRetriever([0]),
            rrf_k=0,
        )


def test_empty_query() -> None:
    hybrid = make_hybrid([0], [0])
    hybrid.index(["retinal disease"])

    with pytest.raises(ValueError, match="Query must not be empty"):
        hybrid.search("   ")


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(top_k: int) -> None:
    hybrid = make_hybrid([0], [0])
    hybrid.index(["retinal disease"])

    with pytest.raises(ValueError, match="top_k must be positive"):
        hybrid.search("retinal", top_k=top_k)


def test_agreement_ranks_passage_first() -> None:
    hybrid = make_hybrid(
        semantic_ranking=[1, 0, 2],
        bm25_ranking=[1, 2, 0],
    )
    hybrid.index(["passage A", "passage B", "passage C"])

    results = hybrid.search("research", top_k=3)

    assert results[0].index == 1
    assert results[0].text == "passage B"


def test_rrf_score_calculation() -> None:
    hybrid = make_hybrid(
        semantic_ranking=[0, 1],
        bm25_ranking=[0, 1],
    )
    hybrid.index(["passage A", "passage B"])

    results = hybrid.search("research", top_k=2)

    assert results[0].score == pytest.approx(2 / 61)
    assert results[1].score == pytest.approx(2 / 62)


def test_top_k_limits_results() -> None:
    hybrid = make_hybrid(
        semantic_ranking=[0, 1, 2],
        bm25_ranking=[2, 1, 0],
    )
    hybrid.index(["A", "B", "C"])

    results = hybrid.search("research", top_k=2)

    assert len(results) == 2


def test_reindex_updates_passages() -> None:
    hybrid = make_hybrid([0], [0])

    hybrid.index(["old passage"])
    hybrid.index(["new passage"])

    results = hybrid.search("new", top_k=1)

    assert results[0].text == "new passage"


def test_preserves_original_indices() -> None:
    hybrid = make_hybrid(
        semantic_ranking=[2, 0, 1],
        bm25_ranking=[2, 1, 0],
    )
    hybrid.index(["A", "B", "C"])

    results = hybrid.search("research", top_k=3)

    assert results[0].index == 2
    assert results[0].text == "C"
