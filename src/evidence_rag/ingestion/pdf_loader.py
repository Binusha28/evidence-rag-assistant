from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm


@dataclass(frozen=True)
class PDFPage:
    source: str
    page_number: int
    text: str


def load_pdf(file_path: str | Path) -> list[PDFPage]:
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Expected a PDF file")

    pages: list[PDFPage] = []

    with pymupdf.open(path) as document:
        if not document.is_pdf:
            raise ValueError("The document is not a PDF")

        page_count = len(document)

    for page_index in range(page_count):
        markdown_text = pymupdf4llm.to_markdown(
            str(path),
            pages=[page_index],
        )

        pages.append(
            PDFPage(
                source=path.name,
                page_number=page_index + 1,
                text=markdown_text.strip(),
            )
        )

    return pages
