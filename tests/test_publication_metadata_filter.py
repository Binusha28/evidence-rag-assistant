"""Regression tests for filtering publication metadata."""

from evidence_rag.ingestion.pipeline import DocumentChunk
from evidence_rag.ingestion.quality_filter import filter_low_information_chunks


def make_chunk(text: str, chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source="research.pdf",
        page_number=1,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_removes_copyright_notice() -> None:
    chunk = make_chunk(
        "COPYRIGHT © 2022 The Authors. This is an open-access "
        "article distributed under the terms of the Creative "
        "Commons Attribution License. No use, distribution or "
        "reproduction is permitted without proper attribution.",
        "research:p1:c1",
    )
    assert filter_low_information_chunks([chunk]) == []


def test_removes_author_affiliation_metadata() -> None:
    chunk = make_chunk(
        "Bin Sheng, Xiaosi Chen, Tingyao Li and Xinyuan Zhang. "
        "1 Department of Computer Science and Engineering, "
        "Shanghai Jiao Tong University. "
        "2 Beijing Tongren Eye Center, Capital Medical University.",
        "research:p1:c2",
    )
    assert filter_low_information_chunks([chunk]) == []


def test_preserves_scientific_findings_with_metadata_terms() -> None:
    chunk = make_chunk(
        "The authors report that deep learning algorithms can "
        "identify diabetic retinopathy using retinal fundus "
        "photographs. The study discusses screening performance "
        "and potential clinical applications.",
        "research:p2:c1",
    )
    assert filter_low_information_chunks([chunk]) == [chunk]


def test_preserves_research_passage_with_copyright_mention() -> None:
    chunk = make_chunk(
        "Copyright restrictions may limit access to training "
        "datasets, but convolutional neural networks can still "
        "be evaluated on publicly available retinal images.",
        "research:p3:c1",
    )
    assert filter_low_information_chunks([chunk]) == [chunk]
