"""Export filtered PDF passages for independent relevance review."""

import json
from pathlib import Path

from evidence_rag.ingestion.pipeline import ingest_pdf
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks
from evidence_rag.ingestion.section_filter import filter_reference_chunks

DATA_DIR = Path("data/processed")
PDF_PATH = Path("data/raw/research.pdf")
OUTPUT_PATH = DATA_DIR / "retrieval_holdout_passages.json"


def main() -> None:
    with (DATA_DIR / "retrieval_holdout_questions.json").open(encoding="utf-8") as file:
        questions = json.load(file)["questions"]

    chunks = ingest_pdf(PDF_PATH)
    chunks = filter_reference_chunks(chunks)
    chunks = filter_low_information_chunks(chunks)

    passages = [
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page_number,
            "text": chunk.text,
        }
        for chunk in chunks
    ]

    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_PATH}")

    output = {
        "questions": questions,
        "passages": passages,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\n=== HELD-OUT JUDGMENT PREPARATION ===")
    print(f"Questions: {len(questions)}")
    print(f"Passages: {len(passages)}")
    print(f"Saved: {OUTPUT_PATH}")
    print("Existing relevance judgments were not modified.")


if __name__ == "__main__":
    main()
