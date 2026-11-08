"""Multi-query BM25 retrieval with reciprocal rank fusion."""

from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.retrieval.bm25_search import (
    BM25Retriever,
    BM25SearchResult,
)


@dataclass(frozen=True)
class FusedSearchResult:
    text: str
    score: float
    index: int


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[BM25SearchResult]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[FusedSearchResult]:
    """Fuse ranked passage lists using reciprocal rank fusion."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    fused_scores: dict[int, float] = {}
    passages: dict[int, str] = {}

    for results in ranked_lists:
        seen_in_list: set[int] = set()

        for rank, result in enumerate(results, start=1):
            if result.index in seen_in_list:
                continue

            seen_in_list.add(result.index)

            fused_scores[result.index] = fused_scores.get(result.index, 0.0) + 1.0 / (rrf_k + rank)
            passages[result.index] = result.text

    ranked_indices = sorted(
        fused_scores,
        key=lambda index: (
            -fused_scores[index],
            index,
        ),
    )[:top_k]

    return [
        FusedSearchResult(
            text=passages[index],
            score=fused_scores[index],
            index=index,
        )
        for index in ranked_indices
    ]


class MultiQueryBM25Retriever:
    """Retrieve and fuse passages for multiple related queries."""

    def __init__(
        self,
        retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        self.retriever = retriever
        self.rrf_k = rrf_k

    def search(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        candidates_per_query: int = 10,
    ) -> list[FusedSearchResult]:
        """Search each query and return fused, deduplicated passages."""

        if not queries:
            raise ValueError("At least one query is required")

        if any(not query.strip() for query in queries):
            raise ValueError("Queries must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if candidates_per_query <= 0:
            raise ValueError("candidates_per_query must be positive")

        ranked_lists = [
            self.retriever.search(
                query=query,
                top_k=candidates_per_query,
            )
            for query in queries
        ]

        return reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
