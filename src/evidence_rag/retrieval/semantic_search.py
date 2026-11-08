from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evidence_rag.retrieval.embeddings import EmbeddingModel


@dataclass(frozen=True)
class SearchResult:
    """A retrieved passage and its similarity score."""

    text: str
    score: float
    index: int


class SemanticRetriever:
    """Search a collection of passages using embeddings."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.passages: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, passages: Sequence[str]) -> None:
        """Generate and store embeddings for passages."""

        if not passages:
            raise ValueError("At least one passage is required")

        self.passages = list(passages)
        self.embeddings = self.embedding_model.encode(self.passages)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return the most semantically similar passages."""

        if self.embeddings is None:
            raise RuntimeError("Index passages before searching")

        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedding_model.encode([query])[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            SearchResult(
                text=self.passages[int(index)],
                score=float(scores[index]),
                index=int(index),
            )
            for index in ranked_indices
        ]
