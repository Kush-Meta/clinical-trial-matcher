"""Match result schemas — per-criterion and trial-level results."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from trial_matcher.schemas.criteria import CriterionType


class MatchStatus(str, Enum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    ABSTAIN = "ABSTAIN"


class AbstentionReason(str, Enum):
    MISSING_DATA = "MISSING_DATA"
    TEMPORAL_AMBIGUITY = "TEMPORAL_AMBIGUITY"
    NOTE_AMBIGUITY = "NOTE_AMBIGUITY"
    CRITERIA_AMBIGUITY = "CRITERIA_AMBIGUITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class EvidenceSnippet(BaseModel):
    text: str
    source_type: Literal["structured", "note_nlp", "computed"]
    source_date: date | None = None
    source_id: str = ""  # lab_id, note_id, etc.
    relevance_score: float = 1.0


class CriterionMatchResult(BaseModel):
    criterion_id: str
    criterion_type: CriterionType
    criterion_summary: str
    raw_criterion_text: str
    status: MatchStatus
    confidence: float = Field(ge=0.0, le=1.0)
    abstention_reason: AbstentionReason | None = None
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    sub_results: list["CriterionMatchResult"] = Field(default_factory=list)
    error_category: str | None = None  # set post-hoc during error analysis

    @property
    def is_blocking(self) -> bool:
        """True if this result blocks trial eligibility."""
        return self.status == MatchStatus.NOT_SATISFIED

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.8:
            return "HIGH"
        if self.confidence >= 0.5:
            return "MEDIUM"
        return "LOW"


CriterionMatchResult.model_rebuild()


class TrialMatchResult(BaseModel):
    patient_id: str
    nct_id: str
    trial_title: str
    overall_eligibility: Literal["ELIGIBLE", "INELIGIBLE", "UNCERTAIN"]
    overall_confidence: float = Field(ge=0.0, le=1.0)

    inclusion_results: list[CriterionMatchResult] = Field(default_factory=list)
    exclusion_results: list[CriterionMatchResult] = Field(default_factory=list)

    # Summary counts
    n_inclusion_satisfied: int = 0
    n_inclusion_abstain: int = 0
    n_exclusion_triggered: int = 0

    blocking_criterion_summary: str | None = None  # first fatal failure
    match_score: float = 0.0  # for ranking

    @property
    def n_inclusion_total(self) -> int:
        return len(self.inclusion_results)

    @property
    def n_exclusion_total(self) -> int:
        return len(self.exclusion_results)

    @property
    def abstention_rate(self) -> float:
        total = self.n_inclusion_total + self.n_exclusion_total
        if total == 0:
            return 0.0
        n_abstain = sum(
            1
            for r in self.inclusion_results + self.exclusion_results
            if r.status == MatchStatus.ABSTAIN
        )
        return n_abstain / total

    def get_all_results(self) -> list[CriterionMatchResult]:
        return self.inclusion_results + self.exclusion_results
