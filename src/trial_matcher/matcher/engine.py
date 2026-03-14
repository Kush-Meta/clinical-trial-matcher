"""Top-level matching engine orchestrator."""

from __future__ import annotations

import logging
from datetime import date

from trial_matcher.config import Settings, get_settings
from trial_matcher.matcher.criterion_evaluators import (
    BehavioralEvaluator,
    BaseCriterionEvaluator,
    CompositeCriterionEvaluator,
    DemographicEvaluator,
    DiagnosisEvaluator,
    FunctionalEvaluator,
    LabValueEvaluator,
    MedicationEvaluator,
    ProcedureEvaluator,
    UnknownEvaluator,
)
from trial_matcher.schemas.criteria import CriterionType, EligibilityCriteriaSet
from trial_matcher.schemas.match import (
    CriterionMatchResult,
    MatchStatus,
    TrialMatchResult,
)
from trial_matcher.schemas.patient import PatientProfile

logger = logging.getLogger(__name__)


def _build_evaluator_map(settings: Settings) -> dict[CriterionType, BaseCriterionEvaluator]:
    base_map: dict[CriterionType, BaseCriterionEvaluator] = {
        CriterionType.LAB_VALUE: LabValueEvaluator(settings),
        CriterionType.DIAGNOSIS: DiagnosisEvaluator(settings),
        CriterionType.MEDICATION: MedicationEvaluator(settings),
        CriterionType.BEHAVIORAL: BehavioralEvaluator(settings),
        CriterionType.DEMOGRAPHIC: DemographicEvaluator(settings),
        CriterionType.FUNCTIONAL: FunctionalEvaluator(settings),
        CriterionType.PROCEDURE: ProcedureEvaluator(settings),
        CriterionType.UNKNOWN: UnknownEvaluator(settings),
    }
    composite = CompositeCriterionEvaluator(settings, base_map)
    base_map[CriterionType.COMPOSITE] = composite
    return base_map


class MatchingEngine:
    """Evaluates patient eligibility against a structured criteria set."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.evaluator_map = _build_evaluator_map(self.settings)

    def match(
        self,
        patient: PatientProfile,
        criteria_set: EligibilityCriteriaSet,
        trial_title: str = "",
    ) -> TrialMatchResult:
        """Run full eligibility evaluation."""
        enrollment_date = patient.enrollment_date

        inclusion_results: list[CriterionMatchResult] = []
        exclusion_results: list[CriterionMatchResult] = []

        # Evaluate inclusion criteria
        for criterion in criteria_set.inclusion:
            criterion_type = criterion.type  # type: ignore[attr-defined]
            evaluator = self.evaluator_map.get(criterion_type, self.evaluator_map[CriterionType.UNKNOWN])
            result = evaluator.evaluate(criterion, patient, enrollment_date)
            inclusion_results.append(result)
            logger.debug("Inclusion [%s] → %s (%.2f)", result.criterion_summary, result.status, result.confidence)

        # Evaluate exclusion criteria
        for criterion in criteria_set.exclusion:
            criterion_type = criterion.type  # type: ignore[attr-defined]
            evaluator = self.evaluator_map.get(criterion_type, self.evaluator_map[CriterionType.UNKNOWN])
            result = evaluator.evaluate(criterion, patient, enrollment_date)
            exclusion_results.append(result)
            logger.debug("Exclusion [%s] → %s (%.2f)", result.criterion_summary, result.status, result.confidence)

        return self._compute_overall(
            patient_id=patient.patient_id,
            nct_id=criteria_set.nct_id,
            trial_title=trial_title,
            inclusion_results=inclusion_results,
            exclusion_results=exclusion_results,
        )

    def _compute_overall(
        self,
        patient_id: str,
        nct_id: str,
        trial_title: str,
        inclusion_results: list[CriterionMatchResult],
        exclusion_results: list[CriterionMatchResult],
    ) -> TrialMatchResult:
        n_inclusion_satisfied = sum(
            1 for r in inclusion_results if r.status == MatchStatus.SATISFIED
        )
        n_inclusion_abstain = sum(
            1 for r in inclusion_results if r.status == MatchStatus.ABSTAIN
        )
        n_exclusion_triggered = sum(
            1 for r in exclusion_results if r.status == MatchStatus.SATISFIED
        )

        # Determine blocking criterion
        blocking_summary: str | None = None

        # Check exclusion first — any triggered → INELIGIBLE
        for r in exclusion_results:
            if r.status == MatchStatus.SATISFIED:
                blocking_summary = f"EXCLUDED: {r.criterion_summary}"
                break

        # Check failed inclusion
        if not blocking_summary:
            for r in inclusion_results:
                if r.status == MatchStatus.NOT_SATISFIED:
                    blocking_summary = f"FAILED INCLUSION: {r.criterion_summary}"
                    break

        # Overall eligibility
        has_failing_inclusion = any(
            r.status == MatchStatus.NOT_SATISFIED for r in inclusion_results
        )
        has_triggered_exclusion = n_exclusion_triggered > 0
        has_abstain = n_inclusion_abstain > 0 or any(
            r.status == MatchStatus.ABSTAIN for r in exclusion_results
        )

        if has_failing_inclusion or has_triggered_exclusion:
            overall = "INELIGIBLE"
        elif has_abstain:
            overall = "UNCERTAIN"
        else:
            overall = "ELIGIBLE"

        # Overall confidence
        all_confidences = [r.confidence for r in inclusion_results + exclusion_results]
        overall_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        # Match score for ranking
        n_total = len(inclusion_results) + len(exclusion_results)
        if n_total > 0:
            n_satisfied = n_inclusion_satisfied
            # Abstained count as 0.5 (generous but not definitive)
            score_numerator = n_satisfied + 0.5 * n_inclusion_abstain
            match_score = (score_numerator / max(1, len(inclusion_results))) * overall_confidence
        else:
            match_score = 0.0

        return TrialMatchResult(
            patient_id=patient_id,
            nct_id=nct_id,
            trial_title=trial_title,
            overall_eligibility=overall,
            overall_confidence=overall_confidence,
            inclusion_results=inclusion_results,
            exclusion_results=exclusion_results,
            n_inclusion_satisfied=n_inclusion_satisfied,
            n_inclusion_abstain=n_inclusion_abstain,
            n_exclusion_triggered=n_exclusion_triggered,
            blocking_criterion_summary=blocking_summary,
            match_score=match_score,
        )
