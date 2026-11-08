from pathlib import Path

import pymupdf
import pytest

from evidence_rag.ingestion.pdf_loader import load_pdf


def create_test_pdf(path: Path, texts: list[str]) -> None:
    """Create a small PDF for testing."""

    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            page.insert_text((72, 72), text)

        document.save(path)


def test_single_page_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "research.pdf"
    create_test_pdf(pdf_path, ["Research requires reliable evidence."])

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0].source == "research.pdf"
    assert pages[0].page_number == 1
    assert pages[0].text == "Research requires reliable evidence."


def test_multiple_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "multi_page.pdf"
    create_test_pdf(
        pdf_path,
        ["First page evidence.", "Second page findings."],
    )

    pages = load_pdf(pdf_path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[0].text == "First page evidence."
    assert pages[1].text == "Second page findings."


def test_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf(missing_path)


def test_invalid_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdf(text_path)


def test_empty_pdf_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty_page.pdf"
    create_test_pdf(pdf_path, [""])

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""
