from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate normalized semantic embeddings for research text."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one normalized embedding per input text."""
        if not texts:
            raise ValueError("At least one text is required")

        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not be empty or whitespace-only")

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=np.float32)
