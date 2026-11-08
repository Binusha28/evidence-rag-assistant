from evidence_rag.ingestion.pipeline import DocumentChunk
from evidence_rag.ingestion.section_filter import filter_reference_chunks


def make_chunk(text: str, chunk_id: str = "test:p1:c0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source="test.pdf",
        page_number=1,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_keeps_research_content() -> None:
    chunks = [
        make_chunk(
            "Deep learning models analyse retinal fundus images "
            "to identify signs of diabetic retinopathy."
        )
    ]

    assert filter_reference_chunks(chunks) == chunks


def test_removes_multiple_doi_references() -> None:
    chunks = [make_chunk("Smith et al. doi: 10.1000/example1. Jones et al. doi: 10.1000/example2.")]

    assert filter_reference_chunks(chunks) == []


def test_removes_bibliography_style_citations() -> None:
    chunks = [make_chunk("Smith A. (2021) 12:123. Jones B. (2022) 15:456.")]

    assert filter_reference_chunks(chunks) == []


def test_preserves_metadata() -> None:
    chunk = make_chunk(
        "Convolutional neural networks can identify retinal lesions.",
        chunk_id="research.pdf:p4:c2",
    )

    result = filter_reference_chunks([chunk])

    assert len(result) == 1
    assert result[0] == chunk
    assert result[0].chunk_id == "research.pdf:p4:c2"


def test_empty_input() -> None:
    assert filter_reference_chunks([]) == []


def test_mixed_content() -> None:
    relevant = make_chunk(
        "AI models can support early diabetic retinopathy screening.",
        chunk_id="research.pdf:p3:c1",
    )

    references = make_chunk(
        "Smith et al. doi: 10.1000/abc. Jones et al. doi: 10.1000/xyz.",
        chunk_id="research.pdf:p13:c1",
    )

    result = filter_reference_chunks([relevant, references])

    assert result == [relevant]
