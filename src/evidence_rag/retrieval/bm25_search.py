import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class BM25SearchResult:
    text: str
    score: float
    index: int


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self) -> None:
        self.passages: list[str] = []
        self.model: BM25Okapi | None = None

    def index(self, passages: Sequence[str]) -> None:
        """Build a BM25 index from text passages."""
        if not passages:
            raise ValueError("At least one passage is required")

        if any(not passage.strip() for passage in passages):
            raise ValueError("Passages must not be empty")

        self.passages = list(passages)

        tokenized_passages = [tokenize(passage) for passage in self.passages]

        self.model = BM25Okapi(tokenized_passages)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[BM25SearchResult]:
        """Return the highest-ranked passages for a query."""
        if self.model is None:
            raise RuntimeError("Index passages before searching")

        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(query)

        scores = self.model.get_scores(query_tokens)

        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )[:top_k]

        return [
            BM25SearchResult(
                text=self.passages[int(index)],
                score=float(scores[index]),
                index=int(index),
            )
            for index in ranked_indices
        ]
