from evidence_rag.verification.citation_placement import (
    check_citation_placement,
)


def test_citation_after_claim_is_valid():
    result = check_citation_placement("CNNs extract retinal image features [E1].")
    assert result.is_valid


def test_citation_at_start_is_invalid():
    result = check_citation_placement("[E2] CNNs detect diabetic retinopathy.")
    assert not result.is_valid
    assert result.leading_citations == ("[E2]",)


def test_citation_at_start_of_second_sentence():
    result = check_citation_placement("CNNs analyse images. [E2] They classify lesions.")
    assert not result.is_valid


def test_multiple_correct_citations():
    result = check_citation_placement("CNNs extract features [E1]. They classify lesions [E2].")
    assert result.is_valid
