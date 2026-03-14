"""Confidence penalty model and abstention threshold."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from trial_matcher.schemas.match import AbstentionReason, MatchStatus

logger = logging.getLogger(__name__)

# Penalty constants
PENALTY_SINGLE_DATA_POINT = 0.10
PENALTY_OLD_LAB = 0.20          # >90 days from enrollment
PENALTY_NOTE_NLP_SOURCE = 0.30  # note-derived vs structured
PENALTY_UNIT_MISMATCH = 0.15    # unit conversion uncertainty
PENALTY_AMBIGUOUS_CRITERION = 0.25
PENALTY_FUZZY_NAME_MATCH = 0.10
PENALTY_NO_INDICATION = 0.20    # medication without indication when required
PENALTY_NEGATION_UNCERTAIN = 0.20  # "rule out X" style negation
PENALTY_MISSING_DATE = 0.15


@dataclass
class ConfidenceModel:
    """Accumulate penalties and compute final confidence."""
    base: float = 1.0
    penalties: list[tuple[str, float]] = field(default_factory=list)

    def penalize(self, reason: str, amount: float) -> "ConfidenceModel":
        self.penalties.append((reason, amount))
        return self

    @property
    def score(self) -> float:
        total_penalty = sum(p for _, p in self.penalties)
        return max(0.0, self.base - total_penalty)

    def should_abstain(self, threshold: float = 0.35) -> tuple[bool, AbstentionReason | None]:
        """Determine if confidence is too low to make a call."""
        if self.score < threshold:
            # Determine primary abstention reason
            if self.penalties:
                # Pick the highest-penalty factor
                worst_reason, _ = max(self.penalties, key=lambda x: x[1])
                reason_map: dict[str, AbstentionReason] = {
                    "missing_data": AbstentionReason.MISSING_DATA,
                    "note_nlp": AbstentionReason.NOTE_AMBIGUITY,
                    "temporal": AbstentionReason.TEMPORAL_AMBIGUITY,
                    "ambiguous_criterion": AbstentionReason.CRITERIA_AMBIGUITY,
                }
                for key, val in reason_map.items():
                    if key in worst_reason.lower():
                        return True, val
            return True, AbstentionReason.LOW_CONFIDENCE
        return False, None

    def __repr__(self) -> str:
        penalty_str = ", ".join(f"{r}=-{p:.2f}" for r, p in self.penalties)
        return f"ConfidenceModel(score={self.score:.2f}, penalties=[{penalty_str}])"


def apply_data_quality_penalties(
    model: ConfidenceModel,
    *,
    source: str = "structured",
    days_old: int = 0,
    n_results: int = 1,
    unit_mismatch: bool = False,
    date_missing: bool = False,
) -> ConfidenceModel:
    """Apply standard data-quality penalties to a confidence model."""
    if source == "note_nlp":
        model.penalize("note_nlp_source", PENALTY_NOTE_NLP_SOURCE)
    if days_old > 90:
        model.penalize("old_lab_result", PENALTY_OLD_LAB)
    if n_results == 1:
        model.penalize("single_data_point", PENALTY_SINGLE_DATA_POINT)
    if unit_mismatch:
        model.penalize("unit_mismatch", PENALTY_UNIT_MISMATCH)
    if date_missing:
        model.penalize("missing_date", PENALTY_MISSING_DATE)
    return model


def compute_composite_confidence(
    operator: str,
    child_confidences: list[float],
    child_statuses: list[MatchStatus],
    n: int | None = None,
) -> float:
    """Compute composite criterion confidence from child results."""
    if not child_confidences:
        return 0.0

    if operator == "AND":
        # Weakest link
        return min(child_confidences)
    elif operator == "OR":
        # Strongest link
        return max(child_confidences)
    elif operator == "NOT":
        return child_confidences[0] if child_confidences else 0.0
    elif operator == "AT_LEAST_N" and n is not None:
        # Mean of top-N confidences
        satisfied_confs = [
            c for c, s in zip(child_confidences, child_statuses)
            if s == MatchStatus.SATISFIED
        ]
        if not satisfied_confs:
            return 0.0
        top_n = sorted(satisfied_confs, reverse=True)[:n]
        return sum(top_n) / len(top_n)
    return float(sum(child_confidences)) / len(child_confidences)
