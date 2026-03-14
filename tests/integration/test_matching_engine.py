"""Integration tests for the matching engine with synthetic patients."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from trial_matcher.config import Settings
from trial_matcher.matcher.engine import MatchingEngine
from trial_matcher.schemas.criteria import (
    BehavioralCriterion,
    DiagnosisCriterion,
    EligibilityCriteriaSet,
    LabValueCriterion,
    MedicationCriterion,
    NumericThreshold,
    TemporalConstraint,
)
from trial_matcher.schemas.match import MatchStatus
from trial_matcher.schemas.patient import (
    ICD10Diagnosis,
    LabResult,
    PatientProfile,
    RxNormMedication,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        demo_mode=True,
        abstention_confidence_threshold=0.35,
        high_confidence_threshold=0.80,
    )


@pytest.fixture
def engine(settings: Settings) -> MatchingEngine:
    return MatchingEngine(settings)


@pytest.fixture
def enrollment_date() -> date:
    return date(2024, 6, 1)


def make_patient(enrollment_date: date, **kwargs: object) -> PatientProfile:
    return PatientProfile(
        patient_id="TEST-001",
        source="synthetic",
        enrollment_date=enrollment_date,
        **kwargs,
    )


def make_criteria_set(
    nct_id: str = "NCT00000000",
    inclusion: list | None = None,
    exclusion: list | None = None,
) -> EligibilityCriteriaSet:
    return EligibilityCriteriaSet(
        nct_id=nct_id,
        raw_text="",
        inclusion=inclusion or [],
        exclusion=exclusion or [],
    )


class TestLabValueMatching:
    def test_satisfied_hba1c_in_range(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            labs=[LabResult(
                lab_name="HbA1c",
                loinc_code="4548-4",
                value=7.5,
                unit="%",
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=30),
                    datetime.min.time(),
                ),
            )],
        )
        criteria = make_criteria_set(
            inclusion=[LabValueCriterion(
                lab_name="HbA1c",
                threshold=NumericThreshold(operator="BETWEEN", value=6.5, value_high=9.5, unit="%"),
                temporal=TemporalConstraint(operator="WITHIN", value=3, unit="months"),
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.inclusion_results[0].status == MatchStatus.SATISFIED

    def test_not_satisfied_hba1c_out_of_range(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            labs=[LabResult(
                lab_name="HbA1c",
                value=5.5,  # Too low
                unit="%",
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=30),
                    datetime.min.time(),
                ),
            )],
        )
        criteria = make_criteria_set(
            inclusion=[LabValueCriterion(
                lab_name="HbA1c",
                threshold=NumericThreshold(operator="BETWEEN", value=6.5, value_high=9.5, unit="%"),
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.inclusion_results[0].status == MatchStatus.NOT_SATISFIED

    def test_abstain_no_lab_data(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(enrollment_date, labs=[])
        criteria = make_criteria_set(
            inclusion=[LabValueCriterion(
                lab_name="HbA1c",
                threshold=NumericThreshold(operator=">", value=7.0, unit="%"),
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.inclusion_results[0].status == MatchStatus.ABSTAIN

    def test_temporal_filter_excludes_old_result(self, engine: MatchingEngine, enrollment_date: date) -> None:
        # Result from 8 months ago — outside 3-month window
        patient = make_patient(
            enrollment_date,
            labs=[LabResult(
                lab_name="HbA1c",
                value=7.5,
                unit="%",
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=240),
                    datetime.min.time(),
                ),
            )],
        )
        criteria = make_criteria_set(
            inclusion=[LabValueCriterion(
                lab_name="HbA1c",
                threshold=NumericThreshold(operator="BETWEEN", value=6.5, value_high=9.5, unit="%"),
                temporal=TemporalConstraint(operator="WITHIN", value=3, unit="months"),
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        # Should abstain — no results in window
        assert result.inclusion_results[0].status == MatchStatus.ABSTAIN


class TestDiagnosisMatching:
    def test_icd10_prefix_match(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            diagnoses=[ICD10Diagnosis(
                icd10_code="I21.9",  # Specific code
                description="Acute MI",
                onset_date=enrollment_date - timedelta(days=60),
                is_active=False,
            )],
        )
        criteria = make_criteria_set(
            inclusion=[DiagnosisCriterion(
                condition_name="myocardial infarction",
                icd10_code="I21",  # Prefix only
                temporal=TemporalConstraint(operator="WITHIN", value=6, unit="months"),
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.inclusion_results[0].status == MatchStatus.SATISFIED

    def test_exclusion_diagnosis_triggers(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            diagnoses=[ICD10Diagnosis(
                icd10_code="E11.9",
                description="Type 2 diabetes",
                is_active=True,
            )],
        )
        criteria = make_criteria_set(
            exclusion=[DiagnosisCriterion(
                condition_name="diabetes mellitus",
                icd10_code="E11",
                required_absent=False,  # Exclude if present
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        # Exclusion triggered → INELIGIBLE
        assert result.n_exclusion_triggered >= 1
        assert result.overall_eligibility == "INELIGIBLE"


class TestMedicationMatching:
    def test_aspirin_81mg_satisfied(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            medications=[RxNormMedication(
                drug_name="aspirin",
                rxnorm_code="1191",
                dose_mg=81,
                indication="MI prevention",
                start_date=enrollment_date - timedelta(days=365),
            )],
        )
        criteria = make_criteria_set(
            inclusion=[MedicationCriterion(
                drug_name="aspirin",
                rxnorm_code="1191",
                dose_mg=81,
                indication="MI prevention",
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.inclusion_results[0].status == MatchStatus.SATISFIED

    def test_medication_not_active(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            medications=[RxNormMedication(
                drug_name="aspirin",
                dose_mg=81,
                start_date=enrollment_date - timedelta(days=365),
                stop_date=enrollment_date - timedelta(days=30),  # Stopped before enrollment
            )],
        )
        criteria = make_criteria_set(
            inclusion=[MedicationCriterion(drug_name="aspirin")]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.inclusion_results[0].status == MatchStatus.NOT_SATISFIED


class TestOverallEligibility:
    def test_eligible_all_satisfied(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            age=55,
            labs=[LabResult(
                lab_name="HbA1c",
                value=7.5,
                unit="%",
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=30), datetime.min.time()
                ),
            )],
            has_drug_abuse=False,
        )
        criteria = make_criteria_set(
            inclusion=[LabValueCriterion(
                lab_name="HbA1c",
                threshold=NumericThreshold(operator="BETWEEN", value=6.5, value_high=9.5),
            )],
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.overall_eligibility == "ELIGIBLE"

    def test_ineligible_exclusion_triggered(self, engine: MatchingEngine, enrollment_date: date) -> None:
        patient = make_patient(
            enrollment_date,
            has_drug_abuse=True,
        )
        criteria = make_criteria_set(
            exclusion=[BehavioralCriterion(
                behavior="DRUG_ABUSE",
                required_absent=False,  # Exclude if present
            )]
        )
        result = engine.match(patient, criteria, "Test Trial")
        assert result.overall_eligibility == "INELIGIBLE"
