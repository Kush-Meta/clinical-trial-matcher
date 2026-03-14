"""Two-pass criteria extractor using Ollama + instructor."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from trial_matcher.config import Settings
from trial_matcher.criteria_parser.prompts import (
    SEGMENTATION_HEADERS,
    SYSTEM_PROMPT,
    build_extraction_prompt,
)
from trial_matcher.schemas.criteria import (
    BehavioralCriterion,
    CompositeCriterion,
    CriterionType,
    DemographicCriterion,
    DiagnosisCriterion,
    EligibilityCriteriaSet,
    FunctionalCriterion,
    LabValueCriterion,
    MedicationCriterion,
    ParsedCriterion,
    ProcedureCriterion,
    UnknownCriterion,
)

logger = logging.getLogger(__name__)

# Type-to-model mapping for instructor validation
_TYPE_MODEL_MAP: dict[str, type] = {
    CriterionType.LAB_VALUE: LabValueCriterion,
    CriterionType.DIAGNOSIS: DiagnosisCriterion,
    CriterionType.MEDICATION: MedicationCriterion,
    CriterionType.PROCEDURE: ProcedureCriterion,
    CriterionType.BEHAVIORAL: BehavioralCriterion,
    CriterionType.FUNCTIONAL: FunctionalCriterion,
    CriterionType.DEMOGRAPHIC: DemographicCriterion,
    CriterionType.COMPOSITE: CompositeCriterion,
    CriterionType.UNKNOWN: UnknownCriterion,
}


class CriteriaExtractor:
    """Two-pass extraction: deterministic segmentation + per-criterion LLM extraction."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import instructor
                import openai

                raw_client = openai.OpenAI(
                    base_url=f"{self.settings.ollama_base_url}/v1",
                    api_key="ollama",
                    timeout=self.settings.ollama_timeout_secs,
                )
                self._client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
            except ImportError as e:
                raise RuntimeError(
                    "Install instructor and openai: pip install instructor openai"
                ) from e
        return self._client

    def extract(self, raw_text: str, nct_id: str) -> EligibilityCriteriaSet:
        """Extract structured criteria from raw eligibility text."""
        incl_lines, excl_lines = self._split_sections(raw_text)
        logger.info(
            "Extracted %d inclusion, %d exclusion lines for %s",
            len(incl_lines),
            len(excl_lines),
            nct_id,
        )

        inclusion: list[ParsedCriterion] = []
        exclusion: list[ParsedCriterion] = []

        for line in incl_lines:
            criterion = self._extract_one(line, "INCLUSION")
            inclusion.append(criterion)

        for line in excl_lines:
            criterion = self._extract_one(line, "EXCLUSION")
            exclusion.append(criterion)

        warnings = [
            getattr(c, "ambiguity_note", "")
            for c in inclusion + exclusion
            if getattr(c, "ambiguous", False)
        ]

        return EligibilityCriteriaSet(
            nct_id=nct_id,
            raw_text=raw_text,
            inclusion=inclusion,
            exclusion=exclusion,
            parse_warnings=[w for w in warnings if w],
        )

    def _extract_one(self, text: str, section: str) -> ParsedCriterion:
        """Extract a single criterion via LLM (with instructor retry)."""
        client = self._get_client()
        prompt = build_extraction_prompt(text, section)

        try:
            # First, get raw dict to determine type
            import json

            response = client.chat.completions.create(
                model=self.settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_model=None,  # raw response first
            )

            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            criterion_type = data.get("type", "UNKNOWN")

            # Validate against the correct Pydantic model
            model_class = _TYPE_MODEL_MAP.get(criterion_type, UnknownCriterion)
            criterion = model_class.model_validate(data)
            return criterion  # type: ignore[return-value]

        except Exception as e:
            logger.warning("Failed to extract criterion '%s': %s", text[:80], e)
            return UnknownCriterion(
                raw_text=text,
                ambiguity_note=f"Extraction failed: {type(e).__name__}",
            )

    # ── Pass 1: deterministic segmentation ──────────────────────────────────

    def _split_sections(self, text: str) -> tuple[list[str], list[str]]:
        """Split raw eligibility text into inclusion and exclusion lines."""
        lines = text.split("\n")
        inclusion_lines: list[str] = []
        exclusion_lines: list[str] = []
        current_section: str | None = None

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            section = self._detect_section_header(stripped)
            if section:
                current_section = section
                continue

            # Skip if we haven't seen a section header yet
            if current_section is None:
                continue

            # Clean up bullet points and numbering
            cleaned = self._clean_line(stripped)
            if not cleaned or len(cleaned) < 5:
                continue

            if current_section == "inclusion":
                inclusion_lines.append(cleaned)
            elif current_section == "exclusion":
                exclusion_lines.append(cleaned)

        return inclusion_lines, exclusion_lines

    def _detect_section_header(self, line: str) -> str | None:
        line_lower = line.lower().rstrip(":").strip()
        for header in SEGMENTATION_HEADERS:
            if line_lower == header or line_lower.startswith(header):
                if "exclusion" in line_lower:
                    return "exclusion"
                return "inclusion"
        return None

    def _clean_line(self, line: str) -> str:
        # Remove leading bullet/number patterns: "1.", "1)", "-", "*", "•"
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", line)
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
        return cleaned.strip()
