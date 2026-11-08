from evidence_rag.ingestion.pipeline import DocumentChunk
from evidence_rag.ingestion.quality_filter import (
    filter_low_information_chunks,
    is_glossary_like,
)


def make_chunk(text: str, chunk_id: str = "test:p1:c0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source="test.pdf",
        page_number=1,
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_detects_abbreviation_glossary() -> None:
    text = (
        "NPDR, Non-proliferative diabetic retinopathy; "
        "OCT, Optical coherence tomography; "
        "PDR, Proliferative diabetic retinopathy; "
        "CNN, Convolutional neural network."
    )

    assert is_glossary_like(text) is True


def test_keeps_scientific_explanation() -> None:
    text = (
        "Convolutional neural networks analyse retinal fundus "
        "images to identify microaneurysms, haemorrhages, "
        "and other indicators of diabetic retinopathy."
    )

    assert is_glossary_like(text) is False


def test_keeps_markdown_table() -> None:
    text = (
        "| Model | Accuracy |\n"
        "| CNN | 95% |\n"
        "NPDR, Non-proliferative; "
        "OCT, Optical coherence tomography; "
        "PDR, Proliferative diabetic retinopathy; "
        "CNN, Convolutional neural network."
    )

    assert is_glossary_like(text) is False


def test_keeps_long_passage() -> None:
    text = (
        "NPDR, Non-proliferative; "
        "OCT, Optical coherence tomography; "
        "PDR, Proliferative diabetic retinopathy; "
        "CNN, Convolutional neural network. "
        + "This research discusses retinal imaging methods. "
        * 15
    )

    assert len(text.split()) > 80
    assert is_glossary_like(text) is False


def test_empty_text() -> None:
    assert is_glossary_like("") is False


def test_filter_removes_glossary() -> None:
    glossary = make_chunk(
        "NPDR, Non-proliferative; "
        "OCT, Optical coherence tomography; "
        "PDR, Proliferative diabetic retinopathy; "
        "CNN, Convolutional neural network.",
        "test:p1:c0",
    )

    research = make_chunk(
        "Deep learning models detect diabetic retinopathy from retinal fundus images.",
        "test:p1:c1",
    )

    filtered = filter_low_information_chunks([glossary, research])

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "test:p1:c1"


def test_preserves_metadata() -> None:
    chunk = make_chunk(
        "Deep learning detects retinal disease.",
        "test:p3:c2",
    )

    filtered = filter_low_information_chunks([chunk])

    assert filtered == [chunk]
    assert filtered[0].page_number == 1


def test_empty_chunk_list() -> None:
    assert filter_low_information_chunks([]) == []
