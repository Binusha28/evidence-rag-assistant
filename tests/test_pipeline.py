from pathlib import Path

import pymupdf
import pytest

from evidence_rag.ingestion.pipeline import ingest_pdf


def create_test_pdf(path: Path, texts: list[str]) -> None:
    """Create a PDF with one page per supplied text."""

    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            page.insert_text((72, 72), text)

        document.save(path)


def test_single_page_ingestion(tmp_path: Path) -> None:
    pdf_path = tmp_path / "research.pdf"
    create_test_pdf(pdf_path, ["one two three four five"])

    chunks = ingest_pdf(pdf_path, chunk_size=3, overlap=1)

    assert len(chunks) == 2
    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four five",
    ]
    assert [chunk.chunk_id for chunk in chunks] == [
        "research.pdf:p1:c0",
        "research.pdf:p1:c1",
    ]
    assert all(chunk.source == "research.pdf" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [
        (0, 3),
        (2, 5),
    ]


def test_multiple_pages_preserve_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    create_test_pdf(
        pdf_path,
        ["alpha beta gamma", "delta epsilon zeta"],
    )

    chunks = ingest_pdf(pdf_path, chunk_size=3, overlap=1)

    assert len(chunks) == 2
    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_id for chunk in chunks] == [
        "study.pdf:p1:c0",
        "study.pdf:p2:c0",
    ]
    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma",
        "delta epsilon zeta",
    ]


def test_empty_page_is_skipped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    create_test_pdf(pdf_path, ["", "useful evidence"])

    chunks = ingest_pdf(pdf_path, chunk_size=3, overlap=1)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].text == "useful evidence"


def test_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_pdf(tmp_path / "missing.pdf")


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_chunk_configuration(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ingest_pdf("unused.pdf", chunk_size=chunk_size, overlap=overlap)
