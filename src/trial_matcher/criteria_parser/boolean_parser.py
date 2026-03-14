"""Boolean logic parser for composite eligibility criteria."""

from __future__ import annotations

import re

from trial_matcher.schemas.criteria import (
    CompositeCriterion,
    DiagnosisCriterion,
    ParsedCriterion,
    UnknownCriterion,
)

# ── Heuristic patterns ────────────────────────────────────────────────────────

AT_LEAST_N_PATTERN = re.compile(
    r"(?:at\s+least|minimum\s+of|(?:two|three|four|five|\d+)\s+or\s+more\s+of)"
    r".*?(?:following|:)",
    re.IGNORECASE | re.DOTALL,
)
AT_LEAST_WORD_TO_N = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "one": 1, "any": 1,
}

ANY_OF_PATTERN = re.compile(r"\bany\s+(?:one\s+)?of\s+the\s+following\b", re.IGNORECASE)
ALL_OF_PATTERN = re.compile(r"\ball\s+of\s+the\s+following\b", re.IGNORECASE)
NOT_PATTERN = re.compile(r"\b(?:not|no|except|excluding)\b", re.IGNORECASE)


def detect_boolean_structure(text: str) -> str | None:
    """Return detected operator or None if no composite structure detected."""
    if AT_LEAST_N_PATTERN.search(text):
        return "AT_LEAST_N"
    if ANY_OF_PATTERN.search(text):
        return "OR"
    if ALL_OF_PATTERN.search(text):
        return "AND"
    return None


def extract_at_least_n(text: str) -> int:
    """Extract N from 'at least N of the following' expressions."""
    # Numeric digit
    m = re.search(r"at\s+least\s+(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Word number
    for word, n in AT_LEAST_WORD_TO_N.items():
        if re.search(rf"\b{word}\b\s+or\s+more", text, re.IGNORECASE):
            return n
    return 2  # sensible default


def split_sub_items(text: str) -> list[str]:
    """Split a multi-item criterion into sub-items."""
    # Try numbered list: "1. ... 2. ... 3. ..."
    numbered = re.split(r"\n?\s*\d+[.)]\s+", text)
    if len(numbered) > 2:
        return [s.strip() for s in numbered if s.strip()]

    # Try bulleted list
    bulleted = re.split(r"\n\s*[-*•]\s+", text)
    if len(bulleted) > 2:
        return [s.strip() for s in bulleted if s.strip()]

    # Try comma/OR separated items (simple heuristic)
    items = re.split(r"\s*,\s*(?:or|and)\s*|\s+or\s+|\s+and\s+", text, flags=re.IGNORECASE)
    if len(items) > 2:
        return [s.strip() for s in items if s.strip()]

    # Semicolon separated
    items = [s.strip() for s in text.split(";") if s.strip()]
    if len(items) > 1:
        return items

    return [text]


def parse_composite(text: str) -> CompositeCriterion | None:
    """Attempt heuristic composite parsing. Returns None if text is not composite."""
    operator = detect_boolean_structure(text)
    if operator is None:
        return None

    sub_items = split_sub_items(text)
    if len(sub_items) <= 1:
        return None

    # Create placeholder UnknownCriterion children (extractor will hydrate these)
    children: list[ParsedCriterion] = [
        UnknownCriterion(raw_text=item, ambiguity_note="sub-item from composite")
        for item in sub_items
    ]

    n: int | None = None
    if operator == "AT_LEAST_N":
        n = extract_at_least_n(text)

    return CompositeCriterion(
        operator=operator,
        n=n,
        children=children,
        raw_text=text,
        ambiguous=False,
    )


def build_not_composite(criterion: ParsedCriterion) -> CompositeCriterion:
    """Wrap a criterion in a NOT composite."""
    return CompositeCriterion(
        operator="NOT",
        children=[criterion],
        raw_text=getattr(criterion, "raw_text", ""),
    )
