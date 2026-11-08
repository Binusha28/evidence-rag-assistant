from dataclasses import dataclass
from pathlib import Path

from evidence_rag.ingestion.chunking import chunk_text
from evidence_rag.ingestion.pdf_loader import load_pdf


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk with source provenance."""

    chunk_id: str
    source: str
    page_number: int
    text: str
    start_word: int
    end_word: int


def ingest_pdf(
    file_path: str | Path,
    chunk_size: int = 100,
    overlap: int = 20,
) -> list[DocumentChunk]:
    """Extract and chunk a PDF while retaining page provenance."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    pages = load_pdf(file_path)
    results: list[DocumentChunk] = []

    for page in pages:
        text_chunks = chunk_text(
            page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(text_chunks):
            results.append(
                DocumentChunk(
                    chunk_id=f"{page.source}:p{page.page_number}:c{index}",
                    source=page.source,
                    page_number=page.page_number,
                    text=chunk.text,
                    start_word=chunk.start_word,
                    end_word=chunk.end_word,
                )
            )

    return results
