"""Regression tests for citation-aware comparison segmentation."""

from evidence_rag.verification.claim_extractor import (
    extract_cited_claims,
)


def test_split_while_comparison_with_independent_citations():
    answer = "CNNs generate feature maps [E1], while Vision Transformers use image patches [E4]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 2
    assert claims[0].text == "CNNs generate feature maps"
    assert claims[0].citation_ids == ("[E1]",)

    assert claims[1].text == ("Vision Transformers use image patches.")
    assert claims[1].citation_ids == ("[E4]",)


def test_split_whereas_comparison():
    answer = "CNNs extract feature maps [E1], whereas ViTs use patch embeddings [E2]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 2
    assert claims[0].citation_ids == ("[E1]",)
    assert claims[1].citation_ids == ("[E2]",)


def test_preserve_single_source_sentence():
    answer = "CNNs generate feature maps [E1]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]",)


def test_preserve_multiple_citations_for_one_claim():
    answer = "Both studies discuss image classification [E1][E2]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == (
        "[E1]",
        "[E2]",
    )


def test_do_not_split_uncited_comparison():
    answer = "CNNs use feature maps while ViTs use patches [E1]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]",)


def test_do_not_split_comparison_without_connector():
    answer = "CNNs generate feature maps [E1], and ViTs use image patches [E2]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 1
    assert claims[0].citation_ids == (
        "[E1]",
        "[E2]",
    )
