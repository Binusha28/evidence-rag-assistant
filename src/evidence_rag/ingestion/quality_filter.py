import re

from evidence_rag.ingestion.pipeline import DocumentChunk


def is_glossary_like(text: str) -> bool:
    """Identify short chunks dominated by abbreviation definitions."""
    words = text.split()

    if not words:
        return False

    # Preserve longer passages containing scientific explanations.
    if len(words) > 80:
        return False

    # Preserve Markdown table content for separate processing.
    if text.lstrip().startswith("|"):
        return False

    # Detect patterns such as "NPDR, Non-proliferative..."
    abbreviation_patterns = re.findall(
        r"\b[A-Za-z][A-Za-z0-9]{1,7},\s+[A-Za-z]",
        text,
    )

    # Glossaries often contain many semicolon-separated definitions.
    definition_count = text.count(";") + 1

    return len(abbreviation_patterns) >= 3 and definition_count >= 4


def is_publication_metadata(text: str) -> bool:
    """Detect publication notices and affiliation-heavy metadata chunks."""
    normalized = " ".join(text.lower().split())

    copyright_notice = ("copyright ©" in normalized or "copyright (c)" in normalized) and (
        "creative commons" in normalized
        or "open-access article" in normalized
        or "reproduction" in normalized
    )

    affiliation_terms = (
        "department of",
        "university",
        "medical university",
        "eye center",
        "research institute",
        "hospital",
    )
    affiliation_count = sum(term in normalized for term in affiliation_terms)
    affiliation_notice = (
        affiliation_count >= 3
        and len(text.split()) <= 100
        and not any(
            term in normalized
            for term in (
                "study discusses",
                "study found",
                "results showed",
                "models detect",
                "algorithms detect",
            )
        )
    )

    return copyright_notice or affiliation_notice


def filter_low_information_chunks(
    chunks: list[DocumentChunk],
) -> list[DocumentChunk]:
    """Exclude glossary-like chunks while retaining other content."""
    return [
        chunk
        for chunk in chunks
        if not is_glossary_like(chunk.text) and not is_publication_metadata(chunk.text)
    ]
