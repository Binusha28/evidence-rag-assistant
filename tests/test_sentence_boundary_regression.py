"""Regression tests for sentence boundaries in generated answers."""

from evidence_rag.verification.claim_extractor import (
    extract_cited_claims,
    extract_uncited_sentences,
    split_sentences,
)


def test_splits_sentences_without_whitespace():
    text = "A CNN generates a feature map [E1].The model achieves 100% accuracy."

    assert split_sentences(text) == [
        "A CNN generates a feature map [E1].",
        "The model achieves 100% accuracy.",
    ]


def test_detects_uncited_sentence_without_whitespace():
    text = "A CNN generates a feature map [E1].The model achieves 100% accuracy."

    assert extract_uncited_sentences(text) == [
        "The model achieves 100% accuracy.",
    ]


def test_does_not_transfer_citation_to_next_sentence():
    text = "A CNN generates a feature map [E1].The model achieves 100% accuracy."

    claims = extract_cited_claims(text)

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]",)
    assert "100% accuracy" not in claims[0].text


def test_preserves_common_abbreviations():
    text = "Dr. Smith analysed the images. The CNN generated a feature map."

    assert split_sentences(text) == [
        "Dr. Smith analysed the images.",
        "The CNN generated a feature map.",
    ]


def test_preserves_decimal_numbers():
    text = "The model achieved 98.5% accuracy. Results were reported."

    assert split_sentences(text) == [
        "The model achieved 98.5% accuracy.",
        "Results were reported.",
    ]
