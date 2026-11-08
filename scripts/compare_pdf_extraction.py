from pathlib import Path

import pymupdf
import pymupdf4llm


def main() -> None:
    pdf_path = Path("data/raw/research.pdf")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as document:
        original_text = document[1].get_text("text", sort=True)

    markdown_text = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=[1],
    )

    print("\nORIGINAL EXTRACTION")
    print("=" * 60)
    print(original_text[:1500])

    print("\nLAYOUT-AWARE MARKDOWN EXTRACTION")
    print("=" * 60)
    print(markdown_text[:1500])


if __name__ == "__main__":
    main()
