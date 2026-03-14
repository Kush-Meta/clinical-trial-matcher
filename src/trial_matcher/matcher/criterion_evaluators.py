"""One evaluator class per CriterionType."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Any

from rapidfuzz import fuzz, process

from trial_matcher.matcher.confidence import (
    PENALTY_AMBIGUOUS_CRITERION,
    PENALTY_FUZZY_NAME_MATCH,
    PENALTY_NEGATION_UNCERTAIN,
    PENALTY_NO_INDICATION,
    ConfidenceModel,
    apply_data_quality_penalties,
)
from trial_matcher.matcher.temporal import days_since, evaluate_in_window, get_most_recent
from trial_matcher.schemas.criteria import (
    BehavioralCriterion,
    CompositeCriterion,
    CriterionType,
    DemographicCriterion,
    DiagnosisCriterion,
    FunctionalCriterion,
    LabValueCriterion,
    MedicationCriterion,
    ParsedCriterion,
    ProcedureCriterion,
    UnknownCriterion,
)
from trial_matcher.schemas.match import (
    AbstentionReason,
    CriterionMatchResult,
    EvidenceSnippet,
    MatchStatus,
)
from trial_matcher.schemas.patient import PatientProfile

logger = logging.getLogger(__name__)

FUZZY_MATCH_THRESHOLD = 85  # rapidfuzz score 0-100


def _make_result(
    criterion: Any,
    status: MatchStatus,
    confidence: float,
    evidence: list[EvidenceSnippet] | None = None,
    abstention_reason: AbstentionReason | None = None,
    sub_results: list[CriterionMatchResult] | None = None,
    summary: str = "",
) -> CriterionMatchResult:
    criterion_type = getattr(criterion, "type", CriterionType.UNKNOWN)
    raw_text = getattr(criterion, "raw_text", "")
    return CriterionMatchResult(
        criterion_id=str(uuid.uuid4())[:8],
        criterion_type=criterion_type,
        criterion_summary=summary or raw_text[:120],
        raw_criterion_text=raw_text,
        status=status,
        confidence=max(0.0, min(1.0, confidence)),
        abstention_reason=abstention_reason,
        evidence=evidence or [],
        sub_results=sub_results or [],
    )


class BaseCriterionEvaluator(ABC):
    def __init__(self, settings: Any) -> None:
        self.settings = settings

    @abstractmethod
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        ...

    def _abstain_missing(self, criterion: Any) -> CriterionMatchResult:
        return _make_result(
            criterion,
            MatchStatus.ABSTAIN,
            0.0,
            abstention_reason=AbstentionReason.MISSING_DATA,
        )


# ── Lab Value ─────────────────────────────────────────────────────────────────


class LabValueEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, LabValueCriterion)
        model = ConfidenceModel()

        if criterion.ambiguous:
            model.penalize("ambiguous_criterion", PENALTY_AMBIGUOUS_CRITERION)

        # Find matching lab results
        labs = patient.get_labs_by_name(criterion.lab_name, criterion.loinc_code)
        if not labs:
            return _make_result(
                criterion,
                MatchStatus.ABSTAIN,
                0.0,
                abstention_reason=AbstentionReason.MISSING_DATA,
                summary=f"{criterion.lab_name} — no data found",
            )

        # Fuzzy match penalty if no LOINC exact match
        if criterion.loinc_code is None or not any(
            l.loinc_code == criterion.loinc_code for l in labs
        ):
            model.penalize("fuzzy_lab_name", PENALTY_FUZZY_NAME_MATCH)

        # Apply temporal filter
        if criterion.temporal:
            if criterion.temporal.operator == "WITHIN":
                window = criterion.temporal.as_timedelta
                if window:
                    cutoff = enrollment_date - window
                    labs = [l for l in labs if l.result_date >= cutoff]
            elif criterion.temporal.operator == "EVER":
                pass  # keep all

        if not labs:
            return _make_result(
                criterion,
                MatchStatus.ABSTAIN,
                model.score * 0.5,
                abstention_reason=AbstentionReason.TEMPORAL_AMBIGUITY,
                summary=f"{criterion.lab_name} — no results in temporal window",
            )

        # Use most recent result
        most_recent = max(labs, key=lambda l: l.result_datetime)
        days_old = days_since(most_recent.result_date, enrollment_date)

        apply_data_quality_penalties(
            model,
            source=most_recent.source,
            days_old=days_old,
            n_results=len(labs),
        )

        # Check unit compatibility
        if criterion.threshold.unit and most_recent.unit:
            if criterion.threshold.unit.lower() != most_recent.unit.lower():
                # Check common conversions (mmol/L vs mg/dL for glucose/HbA1c)
                if not _units_compatible(criterion.threshold.unit, most_recent.unit):
                    model.penalize("unit_mismatch", 0.15)

        satisfied = criterion.threshold.evaluate(most_recent.value)

        should_abstain, abstain_reason = model.should_abstain(
            self.settings.abstention_confidence_threshold
        )
        if should_abstain:
            status = MatchStatus.ABSTAIN
            abstention = abstain_reason
        else:
            status = MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED
            abstention = None

        evidence = [
            EvidenceSnippet(
                text=(
                    f"{most_recent.lab_name}: {most_recent.value} {most_recent.unit} "
                    f"(threshold: {criterion.threshold.operator} {criterion.threshold.value}"
                    f"{criterion.threshold.unit})"
                ),
                source_type=most_recent.source,
                source_date=most_recent.result_date,
                relevance_score=model.score,
            )
        ]

        summary = (
            f"{criterion.lab_name} {criterion.threshold.operator} "
            f"{criterion.threshold.value}{criterion.threshold.unit}"
        )
        if criterion.temporal:
            summary += f" [{criterion.temporal.raw_text}]"

        return _make_result(
            criterion, status, model.score, evidence=evidence,
            abstention_reason=abstention, summary=summary,
        )


def _units_compatible(unit_a: str, unit_b: str) -> bool:
    compatible_groups = [
        {"mg/dl", "mg/dl", "mmol/l", "mmol/l"},
        {"%", "percent"},
        {"g/dl", "g/dl"},
    ]
    a_lower, b_lower = unit_a.lower(), unit_b.lower()
    for group in compatible_groups:
        if a_lower in group and b_lower in group:
            return True
    return a_lower == b_lower


# ── Diagnosis ─────────────────────────────────────────────────────────────────


class DiagnosisEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, DiagnosisCriterion)
        model = ConfidenceModel()

        if criterion.ambiguous:
            model.penalize("ambiguous_criterion", PENALTY_AMBIGUOUS_CRITERION)

        matches = []
        for dx in patient.diagnoses:
            # ICD-10 prefix match
            if criterion.icd10_code and dx.icd10_code:
                code_len = len(criterion.icd10_code.replace(".", ""))
                patient_code_stripped = dx.icd10_code.replace(".", "")[:code_len]
                criterion_code_stripped = criterion.icd10_code.replace(".", "")
                if patient_code_stripped == criterion_code_stripped:
                    matches.append(dx)
                    continue

            # Fuzzy name match fallback
            score = fuzz.token_set_ratio(
                criterion.condition_name.lower(), dx.description.lower()
            )
            if score >= FUZZY_MATCH_THRESHOLD:
                if dx not in matches:
                    matches.append(dx)
                    if dx.source == "note_nlp":
                        model.penalize("note_nlp_source", 0.30)
                    else:
                        model.penalize("fuzzy_name_match", PENALTY_FUZZY_NAME_MATCH)

        # Temporal filtering
        if criterion.temporal and matches:
            filtered = []
            for dx in matches:
                if criterion.temporal.operator == "EVER":
                    filtered.append(dx)
                elif criterion.temporal.operator == "WITHIN" and dx.onset_date:
                    in_window, temp_conf = evaluate_in_window(
                        dx.onset_date, enrollment_date, criterion.temporal
                    )
                    if in_window is not False:
                        filtered.append(dx)
                        if in_window is None:
                            model.penalize("temporal_ambiguity", 0.20)
                else:
                    filtered.append(dx)
            matches = filtered

        # Check for "rule out" mentions in notes (NOTE_AMBIGUITY)
        rule_out = self._check_rule_out(criterion.condition_name, patient)
        if rule_out:
            model.penalize("negation_uncertain", PENALTY_NEGATION_UNCERTAIN)

        found = len(matches) > 0

        if criterion.required_absent:
            satisfied = not found
        else:
            satisfied = found

        if not found and not criterion.required_absent:
            # Check note NLP for mentions
            note_mention = self._check_note_mention(criterion.condition_name, patient)
            if note_mention == "uncertain":
                return _make_result(
                    criterion, MatchStatus.ABSTAIN, model.score * 0.5,
                    abstention_reason=AbstentionReason.NOTE_AMBIGUITY,
                    summary=f"{criterion.condition_name} — uncertain mention in notes",
                )
            # True absence: ABSTAIN if no data at all
            if not patient.diagnoses and not patient.notes:
                return self._abstain_missing(criterion)

        should_abstain, abstain_reason = model.should_abstain(
            self.settings.abstention_confidence_threshold
        )
        if should_abstain and not found:
            status = MatchStatus.ABSTAIN
            abstention = abstain_reason
        else:
            status = MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED
            abstention = None

        evidence = [
            EvidenceSnippet(
                text=f"{dx.icd10_code}: {dx.description} (onset: {dx.onset_date})",
                source_type=dx.source,
                source_date=dx.onset_date,
                relevance_score=model.score,
            )
            for dx in matches[:3]
        ]

        return _make_result(
            criterion, status, model.score, evidence=evidence,
            abstention_reason=abstention,
            summary=f"{criterion.condition_name} ({'absent' if criterion.required_absent else 'present'})",
        )

    def _check_rule_out(self, condition: str, patient: PatientProfile) -> bool:
        for note in patient.notes:
            for entity in note.extracted_entities:
                if (entity.is_uncertain and
                        fuzz.partial_ratio(condition.lower(), entity.text.lower()) >= 70):
                    return True
        return False

    def _check_note_mention(self, condition: str, patient: PatientProfile) -> str:
        """Returns 'present', 'absent', or 'uncertain'."""
        for note in patient.notes:
            for entity in note.extracted_entities:
                score = fuzz.partial_ratio(condition.lower(), entity.text.lower())
                if score >= 70:
                    if entity.is_uncertain:
                        return "uncertain"
                    if entity.is_negated:
                        return "absent"
                    return "present"
        return "absent"


# ── Medication ────────────────────────────────────────────────────────────────


class MedicationEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, MedicationCriterion)
        model = ConfidenceModel()

        if criterion.ambiguous:
            model.penalize("ambiguous_criterion", PENALTY_AMBIGUOUS_CRITERION)

        matches = []
        for med in patient.medications:
            # RxNorm exact match
            if (criterion.rxnorm_code and med.rxnorm_code and
                    criterion.rxnorm_code == med.rxnorm_code):
                if med.active_at(enrollment_date):
                    matches.append(med)
                continue

            # Drug name fuzzy match
            score = fuzz.token_set_ratio(
                criterion.drug_name.lower(), med.drug_name.lower()
            )
            if score >= FUZZY_MATCH_THRESHOLD:
                if criterion.rxnorm_code is None:
                    model.penalize("fuzzy_drug_name", PENALTY_FUZZY_NAME_MATCH)
                if med.active_at(enrollment_date) and med not in matches:
                    matches.append(med)

        # Dose check for aspirin
        if criterion.dose_mg and matches:
            dose_matches = [m for m in matches if m.dose_mg == criterion.dose_mg]
            if not dose_matches and matches:
                model.penalize("dose_unspecified", 0.10)
            else:
                matches = dose_matches

        # Indication check
        if criterion.indication and matches:
            indication_matches = [
                m for m in matches
                if fuzz.partial_ratio(criterion.indication.lower(), m.indication.lower()) >= 70
            ]
            if not indication_matches:
                model.penalize("no_indication", PENALTY_NO_INDICATION)

        # ASP-FOR-MI special logic
        if (criterion.drug_name.lower() in ("aspirin", "asa") and
                criterion.dose_mg == 81 and criterion.indication):
            self._apply_asp_for_mi_logic(criterion, patient, model, matches)

        found = len(matches) > 0
        if criterion.required_absent:
            satisfied = not found
        else:
            satisfied = found

        if not found and not criterion.required_absent and not patient.medications:
            return self._abstain_missing(criterion)

        should_abstain, abstain_reason = model.should_abstain(
            self.settings.abstention_confidence_threshold
        )
        if should_abstain and not found:
            status = MatchStatus.ABSTAIN
            abstention = abstain_reason
        else:
            status = MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED
            abstention = None

        evidence = [
            EvidenceSnippet(
                text=f"{m.drug_name} {m.dose_mg or ''}mg (start: {m.start_date})",
                source_type=m.source,
                source_date=m.start_date,
                relevance_score=model.score,
            )
            for m in matches[:3]
        ]

        return _make_result(
            criterion, status, model.score, evidence=evidence,
            abstention_reason=abstention,
            summary=f"{criterion.drug_name}"
            + (f" {criterion.dose_mg}mg" if criterion.dose_mg else "")
            + (f" for {criterion.indication}" if criterion.indication else ""),
        )

    def _apply_asp_for_mi_logic(
        self, criterion: Any, patient: PatientProfile, model: ConfidenceModel, matches: list
    ) -> None:
        """Special handling for aspirin-for-MI-prevention criterion."""
        has_mi = any(
            dx.icd10_code and dx.icd10_code.startswith("I21")
            for dx in patient.diagnoses
        )
        if not has_mi:
            model.penalize("no_mi_indication_for_aspirin", 0.15)


# ── Behavioral ────────────────────────────────────────────────────────────────


class BehavioralEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, BehavioralCriterion)
        model = ConfidenceModel()

        behavior = criterion.behavior
        required_absent = criterion.required_absent

        # Check structured flags first
        structured_flag: bool | None = None
        if behavior == "DRUG_ABUSE":
            structured_flag = patient.has_drug_abuse
        elif behavior == "ALCOHOL_ABUSE":
            structured_flag = patient.has_alcohol_abuse
        elif behavior == "SMOKING":
            structured_flag = patient.has_smoking_history

        if behavior == "DIETARY_SUPPLEMENT":
            return self._eval_dietary_supplement(criterion, patient, enrollment_date, model)

        if structured_flag is not None:
            found = structured_flag
        else:
            # Fall back to note NLP
            found = self._check_note_behavior(behavior, patient)
            if found is None:
                # Drug/alcohol absence ≠ confirmed absence (patients under-report)
                if behavior in ("DRUG_ABUSE", "ALCOHOL_ABUSE"):
                    return _make_result(
                        criterion,
                        MatchStatus.ABSTAIN,
                        model.score * 0.5,
                        abstention_reason=AbstentionReason.MISSING_DATA,
                        summary=f"{behavior} — under-reporting risk, no structured data",
                    )
                found = False
            else:
                model.penalize("note_nlp_source", 0.30)

        satisfied = (not found) if required_absent else found

        should_abstain, abstain_reason = model.should_abstain(
            self.settings.abstention_confidence_threshold
        )
        status = MatchStatus.ABSTAIN if should_abstain else (
            MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED
        )

        return _make_result(
            criterion, status, model.score,
            abstention_reason=abstain_reason if should_abstain else None,
            summary=f"{behavior} {'absent' if required_absent else 'present'}",
        )

    def _eval_dietary_supplement(
        self, criterion: Any, patient: PatientProfile,
        enrollment_date: date, model: ConfidenceModel,
    ) -> CriterionMatchResult:
        temporal = criterion.temporal
        found = False

        if patient.dietary_supplements:
            if temporal and temporal.operator == "WITHIN" and temporal.as_timedelta:
                # No date on supplements — can't do temporal filtering
                model.penalize("temporal_ambiguity", 0.20)
            found = True
        else:
            # Check notes
            note_found = self._check_note_behavior("DIETARY_SUPPLEMENT", patient)
            if note_found:
                found = True
                model.penalize("note_nlp_source", 0.30)

        satisfied = (not found) if criterion.required_absent else found
        status = MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED

        return _make_result(
            criterion, status, model.score,
            summary=f"DIETSUPP-2MOS {'absent' if criterion.required_absent else 'present'}",
        )

    def _check_note_behavior(self, behavior: str, patient: PatientProfile) -> bool | None:
        behavior_keywords: dict[str, list[str]] = {
            "DRUG_ABUSE": ["drug abuse", "substance abuse", "illicit drug", "drug use disorder"],
            "ALCOHOL_ABUSE": ["alcohol abuse", "alcoholism", "etoh abuse", "aud"],
            "SMOKING": ["smoking", "tobacco", "cigarette", "nicotine"],
            "DIETARY_SUPPLEMENT": ["supplement", "vitamin", "herbal", "omega", "fish oil"],
        }
        keywords = behavior_keywords.get(behavior, [])
        for note in patient.notes:
            for entity in note.extracted_entities:
                for kw in keywords:
                    if kw in entity.text.lower():
                        if entity.is_negated:
                            return False
                        return True
        return None


# ── Demographic ───────────────────────────────────────────────────────────────


class DemographicEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, DemographicCriterion)
        model = ConfidenceModel()

        attr = criterion.attribute

        if attr == "age":
            if patient.age is None:
                return self._abstain_missing(criterion)
            if criterion.threshold is None:
                return _make_result(criterion, MatchStatus.ABSTAIN, 0.0,
                                    abstention_reason=AbstentionReason.CRITERIA_AMBIGUITY)
            satisfied = criterion.threshold.evaluate(patient.age)
            evidence = [EvidenceSnippet(
                text=f"Patient age: {patient.age}",
                source_type="structured",
                relevance_score=1.0,
            )]
            return _make_result(
                criterion,
                MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED,
                1.0, evidence=evidence,
                summary=f"Age {criterion.threshold.operator} {criterion.threshold.value}",
            )

        elif attr == "sex":
            if patient.sex == "Unknown":
                return self._abstain_missing(criterion)
            satisfied = patient.sex == criterion.text_value
            return _make_result(
                criterion,
                MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED,
                1.0,
                summary=f"Sex = {criterion.text_value}",
            )

        elif attr == "bmi":
            bmi_vitals = [v for v in patient.vitals if "bmi" in v.vital_name.lower()]
            if not bmi_vitals:
                return self._abstain_missing(criterion)
            latest = max(bmi_vitals, key=lambda v: v.measured_datetime)
            if criterion.threshold is None:
                return self._abstain_missing(criterion)
            satisfied = criterion.threshold.evaluate(latest.value)
            return _make_result(
                criterion,
                MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED,
                0.9,
                evidence=[EvidenceSnippet(
                    text=f"BMI: {latest.value}",
                    source_type="structured",
                    relevance_score=0.9,
                )],
                summary=f"BMI {criterion.threshold.operator} {criterion.threshold.value}",
            )

        return _make_result(
            criterion, MatchStatus.ABSTAIN, 0.0,
            abstention_reason=AbstentionReason.CRITERIA_AMBIGUITY,
            summary=f"Demographic: {attr} (unsupported)",
        )


# ── Functional ────────────────────────────────────────────────────────────────


class FunctionalEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, FunctionalCriterion)
        # Functional criteria require note NLP or structured assessment
        # Without explicit data, we ABSTAIN
        return _make_result(
            criterion,
            MatchStatus.ABSTAIN,
            0.3,
            abstention_reason=AbstentionReason.MISSING_DATA,
            summary=f"Functional: {criterion.capability[:80]}",
        )


# ── Procedure ─────────────────────────────────────────────────────────────────


class ProcedureEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, ProcedureCriterion)
        model = ConfidenceModel()

        matches = []
        for proc in patient.procedures:
            if criterion.cpt_code and proc.cpt_code == criterion.cpt_code:
                matches.append(proc)
                continue
            score = fuzz.token_set_ratio(
                criterion.procedure_name.lower(), proc.procedure_name.lower()
            )
            if score >= FUZZY_MATCH_THRESHOLD:
                matches.append(proc)
                model.penalize("fuzzy_procedure_name", PENALTY_FUZZY_NAME_MATCH)

        found = len(matches) > 0
        satisfied = (not found) if criterion.required_absent else found

        if not found and not patient.procedures:
            return self._abstain_missing(criterion)

        return _make_result(
            criterion,
            MatchStatus.SATISFIED if satisfied else MatchStatus.NOT_SATISFIED,
            model.score,
            summary=criterion.procedure_name[:80],
        )


# ── Unknown ───────────────────────────────────────────────────────────────────


class UnknownEvaluator(BaseCriterionEvaluator):
    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        return _make_result(
            criterion,
            MatchStatus.ABSTAIN,
            0.0,
            abstention_reason=AbstentionReason.CRITERIA_AMBIGUITY,
            summary="Unknown criterion type",
        )


# ── Composite ─────────────────────────────────────────────────────────────────


class CompositeCriterionEvaluator(BaseCriterionEvaluator):
    def __init__(self, settings: Any, evaluator_map: dict) -> None:
        super().__init__(settings)
        self.evaluator_map = evaluator_map

    def evaluate(
        self,
        criterion: ParsedCriterion,
        patient: PatientProfile,
        enrollment_date: date,
    ) -> CriterionMatchResult:
        assert isinstance(criterion, CompositeCriterion)

        sub_results: list[CriterionMatchResult] = []
        for child in criterion.children:
            child_type = child.type  # type: ignore[attr-defined]
            evaluator = self.evaluator_map.get(child_type)
            if evaluator:
                result = evaluator.evaluate(child, patient, enrollment_date)
            else:
                result = _make_result(
                    child, MatchStatus.ABSTAIN, 0.0,
                    abstention_reason=AbstentionReason.CRITERIA_AMBIGUITY,
                )
            sub_results.append(result)

        op = criterion.operator
        statuses = [r.status for r in sub_results]
        confidences = [r.confidence for r in sub_results]

        if op == "AND":
            if any(s == MatchStatus.NOT_SATISFIED for s in statuses):
                overall = MatchStatus.NOT_SATISFIED
            elif any(s == MatchStatus.ABSTAIN for s in statuses):
                overall = MatchStatus.ABSTAIN
            else:
                overall = MatchStatus.SATISFIED
            conf = min(confidences) if confidences else 0.0

        elif op == "OR":
            if any(s == MatchStatus.SATISFIED for s in statuses):
                overall = MatchStatus.SATISFIED
            elif all(s == MatchStatus.NOT_SATISFIED for s in statuses):
                overall = MatchStatus.NOT_SATISFIED
            else:
                overall = MatchStatus.ABSTAIN
            conf = max(confidences) if confidences else 0.0

        elif op == "NOT":
            if sub_results:
                child_status = sub_results[0].status
                if child_status == MatchStatus.SATISFIED:
                    overall = MatchStatus.NOT_SATISFIED
                elif child_status == MatchStatus.NOT_SATISFIED:
                    overall = MatchStatus.SATISFIED
                else:
                    overall = MatchStatus.ABSTAIN
                conf = confidences[0] if confidences else 0.0
            else:
                overall = MatchStatus.ABSTAIN
                conf = 0.0

        elif op == "AT_LEAST_N" and criterion.n is not None:
            n = criterion.n
            n_satisfied = sum(1 for s in statuses if s == MatchStatus.SATISFIED)
            n_abstain = sum(1 for s in statuses if s == MatchStatus.ABSTAIN)

            if n_satisfied >= n:
                overall = MatchStatus.SATISFIED
            elif n_satisfied + n_abstain < n:
                overall = MatchStatus.NOT_SATISFIED
            else:
                overall = MatchStatus.ABSTAIN

            satisfied_confs = [
                c for c, s in zip(confidences, statuses)
                if s == MatchStatus.SATISFIED
            ]
            conf = sum(sorted(satisfied_confs, reverse=True)[:n]) / n if satisfied_confs else 0.0

        else:
            overall = MatchStatus.ABSTAIN
            conf = 0.0

        abstention = (
            AbstentionReason.MISSING_DATA
            if overall == MatchStatus.ABSTAIN
            else None
        )

        return _make_result(
            criterion, overall, conf,
            sub_results=sub_results,
            abstention_reason=abstention,
            summary=f"{op}({len(criterion.children)} criteria)",
        )
