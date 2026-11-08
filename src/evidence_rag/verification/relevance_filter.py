"""Simple relevance filtering for diabetic retinopathy experiments."""

import re

from evidence_rag.generation.evidence import Evidence


def is_diabetic_retinopathy_relevant(text: str) -> bool:
    """Check for explicit diabetic retinopathy terminology."""
    return bool(
        re.search(
            r"\bdiabetic\s+retinopathy\b|\bDR\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def filter_dr_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Retain passages mentioning diabetic retinopathy."""
    return [item for item in evidence if is_diabetic_retinopathy_relevant(item.text)]
