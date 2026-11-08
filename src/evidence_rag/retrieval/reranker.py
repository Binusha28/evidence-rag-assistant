from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankedResult:
    index: int
    text: str
    score: float


class CrossEncoderReranker:
    """Rerank retrieved passages using a cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not passages:
            return []

        if any(not passage.strip() for passage in passages):
            raise ValueError("Passages must not be empty")

        pairs = [(query, passage) for passage in passages]

        scores = np.asarray(
            self.model.predict(
                pairs,
                show_progress_bar=False,
            )
        ).reshape(-1)

        if len(scores) != len(passages):
            raise ValueError("Model returned an unexpected number of scores")

        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            RerankedResult(
                index=int(index),
                text=passages[int(index)],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]
