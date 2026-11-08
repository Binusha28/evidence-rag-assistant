from evidence_rag.verification.claim_extractor import (
    extract_cited_claims,
    extract_uncited_sentences,
)


def test_single_cited_claim():
    claims = extract_cited_claims("CNNs extract retinal image features [E1].")

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]",)
    assert claims[0].text == "CNNs extract retinal image features."


def test_multiple_citations():
    claims = extract_cited_claims("CNNs support disease classification [E1][E2].")

    assert len(claims) == 1
    assert claims[0].citation_ids == ("[E1]", "[E2]")


def test_uncited_sentence_is_skipped():
    claims = extract_cited_claims("CNNs analyse retinal images.")

    assert claims == []


def test_multiple_cited_sentences():
    claims = extract_cited_claims(
        "CNNs extract features [E1]. They identify candidate lesion regions [E2]."
    )

    assert len(claims) == 2
    assert claims[0].citation_ids == ("[E1]",)
    assert claims[1].citation_ids == ("[E2]",)


def test_uncited_sentence_detected():
    result = extract_uncited_sentences("CNNs analyse retinal images.")

    assert result == ["CNNs analyse retinal images."]


def test_cited_sentence_excluded():
    result = extract_uncited_sentences("CNNs identify retinal lesions [E1].")

    assert result == []


def test_mixed_sentences():
    result = extract_uncited_sentences(
        "CNNs extract image features [E1]. The model achieves 100% accuracy."
    )

    assert result == ["The model achieves 100% accuracy."]
