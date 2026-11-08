"""Tests for scientific certainty requirements in answer generation."""

from evidence_rag.generation.answer_generator import SYSTEM_PROMPT


def test_prompt_preserves_uncertainty() -> None:
    assert "Preserve the certainty and tense" in SYSTEM_PROMPT
    assert "'may'" in SYSTEM_PROMPT
    assert "'could'" in SYSTEM_PROMPT
    assert "'expected'" in SYSTEM_PROMPT


def test_prompt_prohibits_overstated_clinical_benefits() -> None:
    assert (
        "Never present proposed, anticipated, or potential clinical "
        "benefits as experimentally demonstrated outcomes."
    ) in SYSTEM_PROMPT


def test_prompt_requires_narrow_supported_claims() -> None:
    assert "narrowest scientifically meaningful claim" in SYSTEM_PROMPT
    assert "directly supported by its cited passage" in SYSTEM_PROMPT
