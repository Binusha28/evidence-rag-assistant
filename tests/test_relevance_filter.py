from evidence_rag.verification.relevance_filter import (
    is_diabetic_retinopathy_relevant,
)


def test_full_disease_name():
    assert is_diabetic_retinopathy_relevant(
        "CNN models detect diabetic retinopathy from retinal images."
    )


def test_abbreviation():
    assert is_diabetic_retinopathy_relevant("The model classifies DR severity.")


def test_unrelated_condition():
    assert not is_diabetic_retinopathy_relevant("CNNs diagnose central serous chorioretinopathy.")


def test_other_ophthalmic_condition():
    assert not is_diabetic_retinopathy_relevant("The model detects epiretinal membranes.")
