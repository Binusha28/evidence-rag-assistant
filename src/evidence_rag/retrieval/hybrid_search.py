from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.retrieval.bm25_search import BM25Retriever
from evidence_rag.retrieval.semantic_search import SemanticRetriever


@dataclass(frozen=True)
class HybridSearchResult:
    text: str
    score: float
    index: int


class HybridRetriever:
    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.passages: list[str] = []

    def index(self, passages: Sequence[str]) -> None:
        if not passages:
            raise ValueError("At least one passage is required")

        new_passages = list(passages)

        self.semantic_retriever.index(new_passages)
        self.bm25_retriever.index(new_passages)

        self.passages = new_passages

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[HybridSearchResult]:
        if not self.passages:
            raise RuntimeError("Index passages before searching")

        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_count = min(
            len(self.passages),
            max(top_k * 3, 10),
        )

        semantic_results = self.semantic_retriever.search(
            query,
            top_k=candidate_count,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=candidate_count,
        )

        fused_scores: dict[int, float] = {}

        for results in (semantic_results, bm25_results):
            for rank, result in enumerate(results, start=1):
                fused_scores[result.index] = fused_scores.get(result.index, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        ranked_indices = sorted(
            fused_scores,
            key=lambda index: (-fused_scores[index], index),
        )[:top_k]

        return [
            HybridSearchResult(
                text=self.passages[index],
                score=fused_scores[index],
                index=index,
            )
            for index in ranked_indices
        ]
