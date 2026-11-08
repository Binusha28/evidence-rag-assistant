import re

from evidence_rag.ingestion.pipeline import DocumentChunk


def filter_reference_chunks(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """Remove chunks that appear to contain bibliography entries."""

    filtered: list[DocumentChunk] = []

    for chunk in chunks:
        text = chunk.text

        doi_count = len(re.findall(r"\bdoi\s*:", text, flags=re.IGNORECASE))
        citation_count = len(re.findall(r"\b(?:19|20)\d{2}\)\s+\d+:", text))

        if doi_count >= 2 or citation_count >= 2:
            continue

        filtered.append(chunk)

    return filtered
