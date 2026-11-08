from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.section_filter import filter_reference_chunks

TARGET_CHUNK_IDS = [
    "research.pdf:p11:c3",
    "research.pdf:p2:c8",
]


def main() -> None:
    chunks = filter_reference_chunks(ingest_pdf("data/raw/research.pdf"))

    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}

    for chunk_id in TARGET_CHUNK_IDS:
        print("\n" + "=" * 70)
        print(f"CHUNK: {chunk_id}")

        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            print("Chunk not found")
            continue

        print(f"PAGE: {chunk.page_number}")
        print(f"WORDS: {len(chunk.text.split())}")
        print(f"START WORD: {chunk.start_word}")
        print(f"END WORD: {chunk.end_word}")
        print("\nFULL TEXT:")
        print(chunk.text)


if __name__ == "__main__":
    main()
