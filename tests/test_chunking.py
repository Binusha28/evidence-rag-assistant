import pytest

from evidence_rag.ingestion.chunking import chunk_text


def test_short_text():
    chunks = chunk_text(
        "Research requires reliable evidence.",
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Research requires reliable evidence."
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 4

    assert len(chunks) == 1
    assert chunks[0].text == "Research requires reliable evidence."
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 4


def test_overlapping_chunks():
    text = "one two three four five six seven"
    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six seven",
    ]


def test_empty_text():
    assert chunk_text("   ") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("sample text", chunk_size, overlap)
