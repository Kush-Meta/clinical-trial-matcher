"""Post-hoc error taxonomy classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trial_matcher.schemas.match import AbstentionReason, CriterionMatchResult, MatchStatus


class ErrorCategory(str, Enum):
    CRITERIA_PARSING_ERROR = "CRITERIA_PARSING_ERROR"
    TEMPORAL_FAILURE = "TEMPORAL_FAILURE"
    NEGATION_ERROR = "NEGATION_ERROR"
    PATIENT_DATA_GAP = "PATIENT_DATA_GAP"
    THRESHOLD_ERROR = "THRESHOLD_ERROR"
    COMPOSITE_LOGIC_ERROR = "COMPOSITE_LOGIC_ERROR"
    CORRECT = "CORRECT"
    ABSTAINED = "ABSTAINED"


@dataclass
class ClassifiedError:
    criterion_summary: str
    criterion_type: str
    prediction: str
    ground_truth: str
    confidence: float
    error_category: ErrorCategory
    abstention_reason: str | None = None


def classify_error(
    result: CriterionMatchResult,
    ground_truth_label: int,  # 1 or 0
) -> ClassifiedError:
    """
    Classify a prediction error into the taxonomy.

    ground_truth_label: 1 = met, 0 = not met
    """
    pred_label: int
    if result.status == MatchStatus.SATISFIED:
        pred_label = 1
    elif result.status == MatchStatus.NOT_SATISFIED:
        pred_label = 0
    else:  # ABSTAIN
        pred_label = -1

    # Determine error category
    if pred_label == -1:
        category = ErrorCategory.ABSTAINED
    elif pred_label == ground_truth_label:
        category = ErrorCategory.CORRECT
    elif getattr(result, "error_category", None):
        # Already classified
        try:
            category = ErrorCategory(result.error_category)
        except ValueError:
            category = _infer_category(result)
    else:
        category = _infer_category(result)

    return ClassifiedError(
        criterion_summary=result.criterion_summary,
        criterion_type=result.criterion_type.value,
        prediction=result.status.value,
        ground_truth="met" if ground_truth_label == 1 else "not met",
        confidence=result.confidence,
        error_category=category,
        abstention_reason=result.abstention_reason.value if result.abstention_reason else None,
    )


def _infer_category(result: CriterionMatchResult) -> ErrorCategory:
    """Infer error category from result metadata."""
    # Check abstention reason
    if result.abstention_reason == AbstentionReason.TEMPORAL_AMBIGUITY:
        return ErrorCategory.TEMPORAL_FAILURE
    if result.abstention_reason == AbstentionReason.NOTE_AMBIGUITY:
        return ErrorCategory.NEGATION_ERROR
    if result.abstention_reason == AbstentionReason.MISSING_DATA:
        return ErrorCategory.PATIENT_DATA_GAP
    if result.abstention_reason == AbstentionReason.CRITERIA_AMBIGUITY:
        return ErrorCategory.CRITERIA_PARSING_ERROR

    # Check criterion type for composite errors
    from trial_matcher.schemas.criteria import CriterionType
    if result.criterion_type == CriterionType.COMPOSITE and result.sub_results:
        return ErrorCategory.COMPOSITE_LOGIC_ERROR

    # Default: threshold/classification error
    return ErrorCategory.THRESHOLD_ERROR


def build_error_taxonomy(
    results: list[CriterionMatchResult],
    ground_truths: list[int],
) -> dict[ErrorCategory, list[ClassifiedError]]:
    """Classify all results and group by error category."""
    taxonomy: dict[ErrorCategory, list[ClassifiedError]] = {
        cat: [] for cat in ErrorCategory
    }

    for result, gt in zip(results, ground_truths):
        classified = classify_error(result, gt)
        taxonomy[classified.error_category].append(classified)

    return taxonomy


def taxonomy_summary(
    taxonomy: dict[ErrorCategory, list[ClassifiedError]],
) -> dict[str, int]:
    """Return count per error category."""
    return {cat.value: len(items) for cat, items in taxonomy.items()}
