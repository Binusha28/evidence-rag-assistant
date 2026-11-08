from collections.abc import Sequence
from dataclasses import dataclass

from evidence_rag.ingestion.pipeline import DocumentChunk
from evidence_rag.retrieval.semantic_search import SemanticRetriever


@dataclass(frozen=True)
class CitedSearchResult:
    """A retrieved passage with its PDF provenance."""

    chunk_id: str
    source: str
    page_number: int
    text: str
    score: float
    start_word: int
    end_word: int


class DocumentSearcher:
    """Search source-aware document chunks."""

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever
        self.chunks: list[DocumentChunk] = []

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Index document chunks and retain their metadata."""

        if not chunks:
            raise ValueError("At least one document chunk is required")

        new_chunks = list(chunks)
        self.retriever.index([chunk.text for chunk in new_chunks])
        self.chunks = new_chunks

    def search(self, query: str, top_k: int = 3) -> list[CitedSearchResult]:
        """Return ranked passages with source metadata."""

        results = self.retriever.search(query, top_k=top_k)

        return [
            CitedSearchResult(
                chunk_id=self.chunks[result.index].chunk_id,
                source=self.chunks[result.index].source,
                page_number=self.chunks[result.index].page_number,
                text=result.text,
                score=result.score,
                start_word=self.chunks[result.index].start_word,
                end_word=self.chunks[result.index].end_word,
            )
            for result in results
        ]
