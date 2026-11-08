from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A text segment with its word positions."""

    text: str
    start_word: int
    end_word: int


def chunk_text(
    text: str,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[TextChunk]:
    """Split text into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap value")

    words = text.split()

    if not words:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))

        chunks.append(
            TextChunk(
                text=" ".join(words[start:end]),
                start_word=start,
                end_word=end,
            )
        )

        if end == len(words):
            break

    return chunks
