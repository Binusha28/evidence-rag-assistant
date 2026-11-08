from collections import Counter

from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.section_filter import filter_reference_chunks


def main() -> None:
    chunks = ingest_pdf("data/raw/research.pdf")
    filtered = filter_reference_chunks(chunks)

    retained_ids = {chunk.chunk_id for chunk in filtered}
    removed = [chunk for chunk in chunks if chunk.chunk_id not in retained_ids]

    print("\nFILTER ANALYSIS")
    print("=" * 60)
    print(f"Original chunks: {len(chunks)}")
    print(f"Retained chunks: {len(filtered)}")
    print(f"Removed chunks: {len(removed)}")

    print("\nREMOVED CHUNKS BY PAGE")
    print("=" * 60)

    page_counts = Counter(chunk.page_number for chunk in removed)

    for page, count in sorted(page_counts.items()):
        print(f"Page {page}: {count} removed")

    print("\nSAMPLE REMOVED CHUNKS")
    print("=" * 60)

    for chunk in removed[:8]:
        print(f"\nChunk: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Text: {chunk.text[:350]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
