"""Two-pass criteria extractor using Groq (cloud) or Ollama (local) + instructor."""

from __future__ import annotations

import json
import logging
import re
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


def _build_openai_client(base_url: str, api_key: str, timeout: int) -> Any:
    """Build an instructor-wrapped OpenAI-compatible client."""
    try:
        import instructor
        import openai

        raw = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        return instructor.from_openai(raw, mode=instructor.Mode.JSON)
    except ImportError as e:
        raise RuntimeError(
            "Install instructor and openai: pip install instructor openai"
        ) from e


class CriteriaExtractor:
    """
    Two-pass extraction: deterministic segmentation + per-criterion LLM extraction.

    Backends (in priority order):
      1. Groq  — free cloud API, llama-3.1-8b-instant
      2. Ollama — local, llama3.1:8b
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None
        self._model: str = ""

    def _get_client(self) -> tuple[Any, str]:
        if self._client is not None:
            return self._client, self._model

        s = self.settings

        if s.llm_backend == "groq" and s.groq_api_key:
            logger.info("Using Groq backend (%s)", s.groq_model)
            self._client = _build_openai_client(
                s.groq_base_url, s.groq_api_key, timeout=60
            )
            self._model = s.groq_model

        else:
            # Fallback to Ollama
            logger.info("Using Ollama backend (%s)", s.ollama_model)
            self._client = _build_openai_client(
                f"{s.ollama_base_url}/v1", "ollama", timeout=s.ollama_timeout_secs
            )
            self._model = s.ollama_model

        return self._client, self._model

    def extract(self, raw_text: str, nct_id: str) -> EligibilityCriteriaSet:
        """Extract structured criteria from raw eligibility text."""
        incl_lines, excl_lines = self._split_sections(raw_text)
        logger.info(
            "%s: %d inclusion, %d exclusion lines", nct_id, len(incl_lines), len(excl_lines)
        )

        inclusion: list[ParsedCriterion] = [
            self._extract_one(line, "INCLUSION") for line in incl_lines
        ]
        exclusion: list[ParsedCriterion] = [
            self._extract_one(line, "EXCLUSION") for line in excl_lines
        ]

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
        client, model = self._get_client()
        prompt = build_extraction_prompt(text, section)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_model=None,
            )
            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            criterion_type = data.get("type", "UNKNOWN")
            model_class = _TYPE_MODEL_MAP.get(criterion_type, UnknownCriterion)
            return model_class.model_validate(data)  # type: ignore[return-value]

        except Exception as e:
            logger.warning("Extraction failed for '%s': %s", text[:60], e)
            return UnknownCriterion(
                raw_text=text,
                ambiguity_note=f"Extraction failed: {type(e).__name__}",
            )

    # ── Pass 1: deterministic segmentation ──────────────────────────────────

    def _split_sections(self, text: str) -> tuple[list[str], list[str]]:
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
            if current_section is None:
                continue
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
                return "exclusion" if "exclusion" in line_lower else "inclusion"
        return None

    def _clean_line(self, line: str) -> str:
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", line)
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
        return cleaned.strip()
