"""Clinical note NLP pipeline using medspaCy + negspaCy."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from trial_matcher.schemas.patient import ClinicalNote, ExtractedEntity

logger = logging.getLogger(__name__)

# Section keywords for MIMIC discharge summaries
SECTION_PATTERNS = {
    "history_of_present_illness": [
        "history of present illness", "hpi", "presenting complaint"
    ],
    "past_medical_history": [
        "past medical history", "pmh", "medical history", "past history"
    ],
    "medications": [
        "medications", "current medications", "home medications", "medication list"
    ],
    "social_history": [
        "social history", "sh", "social"
    ],
    "family_history": [
        "family history", "fh"
    ],
    "review_of_systems": [
        "review of systems", "ros"
    ],
    "assessment_and_plan": [
        "assessment and plan", "assessment", "plan", "impression"
    ],
    "labs": [
        "laboratory", "labs", "lab results", "pertinent results"
    ],
}

RULE_OUT_PATTERNS = re.compile(
    r"\b(?:rule\s+out|r/o|?:probable|possible|suspected|questionable|query)\b",
    re.IGNORECASE,
)

NEGATION_PATTERNS = re.compile(
    r"\b(?:no|not|without|denies|denied|negative\s+for|absent|never|none)\b",
    re.IGNORECASE,
)

HISTORY_PATTERNS = re.compile(r"\bhist(?:ory)?\s+of\b", re.IGNORECASE)


def detect_section(text: str, position: int) -> str:
    """Detect which section of a note a character position falls in."""
    text_before = text[:position].lower()
    last_section = "unknown"

    for section_name, keywords in SECTION_PATTERNS.items():
        for kw in keywords:
            idx = text_before.rfind(kw)
            if idx >= 0:
                last_section = section_name
                break

    return last_section


class NoteNLPPipeline:
    """
    NLP pipeline for clinical notes.
    Uses medspaCy + negspaCy when available; falls back to regex.
    """

    def __init__(self) -> None:
        self._nlp: Any = None
        self._medspacy_available = False
        self._try_load_medspacy()

    def _try_load_medspacy(self) -> None:
        try:
            import medspacy  # type: ignore
            import spacy

            self._nlp = medspacy.load()
            self._medspacy_available = True
            logger.info("medspaCy pipeline loaded successfully")
        except ImportError:
            logger.info("medspaCy not available — using regex fallback")

    def process_note(self, note: ClinicalNote) -> ClinicalNote:
        """Extract entities from a clinical note, return updated note."""
        if self._medspacy_available:
            return self._process_with_medspacy(note)
        return self._process_with_regex(note)

    def _process_with_medspacy(self, note: ClinicalNote) -> ClinicalNote:
        """Full medspaCy pipeline with negation and section detection."""
        doc = self._nlp(note.text)
        entities = []

        for ent in doc.ents:
            is_negated = getattr(ent._, "is_negated", False)
            is_uncertain = RULE_OUT_PATTERNS.search(
                note.text[max(0, ent.start_char - 30): ent.start_char]
            ) is not None

            section = detect_section(note.text, ent.start_char)

            entities.append(ExtractedEntity(
                text=ent.text,
                entity_type=ent.label_,
                is_negated=is_negated,
                is_uncertain=is_uncertain,
                section=section,
                confidence=0.85,
            ))

        return note.model_copy(update={"extracted_entities": entities})

    def _process_with_regex(self, note: ClinicalNote) -> ClinicalNote:
        """Regex-based fallback pipeline."""
        entities = []
        text = note.text

        # Split into sentences
        sentences = re.split(r"[.!?]\s+|\n", text)

        for sent in sentences:
            if not sent.strip():
                continue

            is_negated = bool(NEGATION_PATTERNS.search(sent[:50]))
            is_uncertain = bool(RULE_OUT_PATTERNS.search(sent[:80]))

            # Simple entity extraction: capitalized medical terms
            # (real pipeline uses scispaCy NER)
            med_terms = re.findall(
                r"\b[A-Z][a-zA-Z]{3,}\b(?:\s+[a-z][a-zA-Z]{2,})?",
                sent
            )
            for term in med_terms:
                entities.append(ExtractedEntity(
                    text=term,
                    entity_type="UNKNOWN",
                    is_negated=is_negated,
                    is_uncertain=is_uncertain,
                    section="unknown",
                    confidence=0.5,
                ))

        return note.model_copy(update={"extracted_entities": entities})
