"""Regression tests for scientific sentence segmentation."""

from evidence_rag.verification.claim_extractor import (
    extract_cited_claims,
    extract_uncited_sentences,
    split_sentences,
)


def test_et_al_does_not_split_sentence():
    answer = "Ryu et al. evaluated OCTA images [E2]."

    assert split_sentences(answer) == [answer]


def test_dr_abbreviation_does_not_split_sentence():
    answer = "Dr. Smith studied retinal images [E1]."

    assert split_sentences(answer) == [answer]


def test_two_cited_sentences():
    answer = "Ryu et al. evaluated OCTA images [E2]. CNNs identify lesion regions [E1]."

    claims = extract_cited_claims(answer)

    assert len(claims) == 2
    assert claims[0].citation_ids == ("[E2]",)
    assert claims[1].citation_ids == ("[E1]",)


def test_no_false_uncited_sentence():
    answer = "Ryu et al. evaluated OCTA images [E2]."

    assert extract_uncited_sentences(answer) == []


def test_genuine_uncited_sentence_detected():
    answer = "CNNs identify candidate lesions [E1]. This method is always perfect."

    assert extract_uncited_sentences(answer) == ["This method is always perfect."]


def test_empty_input():
    assert split_sentences("") == []
    assert extract_cited_claims("") == []
    assert extract_uncited_sentences("") == []
