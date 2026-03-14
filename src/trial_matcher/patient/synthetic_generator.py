"""Synthetic patient generator with n2c2 prevalence distributions."""

from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from trial_matcher.schemas.patient import (
    ICD10Diagnosis,
    LabResult,
    PatientProfile,
    RxNormMedication,
    VitalSign,
)

# n2c2 2018 criteria prevalence distributions
N2C2_CRITERIA_PREVALENCE: dict[str, float] = {
    "HBA1C": 0.65,        # HbA1c 6.5–9.5%
    "MI_6MOS": 0.10,      # MI within 6 months
    "ADVANCED_CAD": 0.40, # Advanced coronary artery disease
    "ASP_FOR_MI": 0.55,   # Aspirin 81mg for MI prevention
    "CREATININE": 0.35,   # Elevated creatinine
    "DRUG_ABUSE": 0.08,
    "ALCOHOL_ABUSE": 0.15,
    "MAKES_DECISIONS": 0.90,  # Most patients make own decisions
    "ENGLISH": 0.85,
    "ABDOMINAL": 0.25,
    "MAJOR_DIABETES": 0.60,
    "KETO_1YR": 0.05,     # Ketoacidosis in past year (rare)
    "DIETSUPP_2MOS": 0.35,
}

# Common ICD-10 codes
_ICD_CODES = {
    "type2_diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "mi": ("I21.9", "Acute myocardial infarction, unspecified"),
    "cad": ("I25.10", "Atherosclerotic heart disease of native coronary artery"),
    "ckd_3": ("N18.3", "Chronic kidney disease, stage 3 (moderate)"),
    "afib": ("I48.91", "Unspecified atrial fibrillation"),
    "heart_failure": ("I50.9", "Heart failure, unspecified"),
    "hyperlipidemia": ("E78.5", "Hyperlipidemia, unspecified"),
    "obesity": ("E66.9", "Obesity, unspecified"),
    "copd": ("J44.1", "COPD with acute exacerbation"),
    "ketoacidosis": ("E11.10", "T2DM with diabetic ketoacidosis without coma"),
    "peripheral_vascular": ("I73.9", "Peripheral vascular disease, unspecified"),
}

# Common medications
_MEDICATIONS = {
    "metformin": ("metformin", "860974", 1000),
    "aspirin_81": ("aspirin", "1191", 81),
    "lisinopril": ("lisinopril", "29046", 10),
    "atorvastatin": ("atorvastatin", "83367", 40),
    "metoprolol": ("metoprolol", "41493", 50),
    "insulin": ("insulin glargine", "274783", None),
    "glipizide": ("glipizide", "25789", 5),
    "amlodipine": ("amlodipine", "17767", 5),
}


def _random_date_before(ref: date, max_days: int) -> date:
    days_back = random.randint(1, max_days)
    return ref - timedelta(days=days_back)


def _random_date_around(ref: date, window_days: int = 30) -> date:
    offset = random.randint(-window_days, window_days)
    return ref + timedelta(days=offset)


class SyntheticPatientGenerator:
    """Generate realistic PatientProfile objects for demo/testing."""

    def __init__(self, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        n: int = 1,
        enrollment_date: date | None = None,
    ) -> list[PatientProfile]:
        if enrollment_date is None:
            enrollment_date = date.today() - timedelta(days=random.randint(30, 365))
        return [self._generate_one(enrollment_date) for _ in range(n)]

    def _generate_one(self, enrollment_date: date) -> PatientProfile:
        patient_id = f"SYN-{uuid.uuid4().hex[:8].upper()}"
        age = random.randint(45, 80)
        sex = random.choice(["M", "F"])

        # Baseline comorbidities
        diagnoses = self._generate_diagnoses(enrollment_date)
        medications = self._generate_medications(enrollment_date, diagnoses)
        labs = self._generate_labs(enrollment_date, diagnoses)
        vitals = self._generate_vitals(enrollment_date)

        # Behavioral
        has_drug_abuse = random.random() < N2C2_CRITERIA_PREVALENCE["DRUG_ABUSE"]
        has_alcohol_abuse = random.random() < N2C2_CRITERIA_PREVALENCE["ALCOHOL_ABUSE"]
        has_dietary_supp = random.random() < N2C2_CRITERIA_PREVALENCE["DIETSUPP_2MOS"]
        dietary_supplements = (
            random.sample(["fish oil", "vitamin D", "multivitamin", "CoQ10", "magnesium"], 2)
            if has_dietary_supp
            else []
        )

        return PatientProfile(
            patient_id=patient_id,
            source="synthetic",
            age=age,
            sex=sex,
            enrollment_date=enrollment_date,
            diagnoses=diagnoses,
            medications=medications,
            labs=labs,
            vitals=vitals,
            has_drug_abuse=has_drug_abuse,
            has_alcohol_abuse=has_alcohol_abuse,
            has_smoking_history=random.random() < 0.25,
            dietary_supplements=dietary_supplements,
            data_completeness_score=random.uniform(0.7, 1.0),
        )

    def _generate_diagnoses(self, enrollment_date: date) -> list[ICD10Diagnosis]:
        diagnoses = []

        # Type 2 diabetes — most patients in this cohort
        if random.random() < N2C2_CRITERIA_PREVALENCE["MAJOR_DIABETES"]:
            onset = _random_date_before(enrollment_date, 365 * 10)
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["type2_diabetes"][0],
                description=_ICD_CODES["type2_diabetes"][1],
                onset_date=onset,
                is_active=True,
            ))

        # Hypertension — very common
        if random.random() < 0.70:
            onset = _random_date_before(enrollment_date, 365 * 8)
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["hypertension"][0],
                description=_ICD_CODES["hypertension"][1],
                onset_date=onset,
                is_active=True,
            ))

        # MI within 6 months
        if random.random() < N2C2_CRITERIA_PREVALENCE["MI_6MOS"]:
            days_back = random.randint(1, 180)
            onset = enrollment_date - timedelta(days=days_back)
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["mi"][0],
                description=_ICD_CODES["mi"][1],
                onset_date=onset,
                is_active=False,
                resolved_date=onset + timedelta(days=random.randint(5, 30)),
            ))

        # Advanced CAD
        if random.random() < N2C2_CRITERIA_PREVALENCE["ADVANCED_CAD"]:
            onset = _random_date_before(enrollment_date, 365 * 5)
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["cad"][0],
                description=_ICD_CODES["cad"][1],
                onset_date=onset,
                is_active=True,
            ))

        # CKD — ties to creatinine
        if random.random() < 0.30:
            onset = _random_date_before(enrollment_date, 365 * 3)
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["ckd_3"][0],
                description=_ICD_CODES["ckd_3"][1],
                onset_date=onset,
                is_active=True,
            ))

        # Ketoacidosis (rare)
        if random.random() < N2C2_CRITERIA_PREVALENCE["KETO_1YR"]:
            days_back = random.randint(30, 365)
            onset = enrollment_date - timedelta(days=days_back)
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["ketoacidosis"][0],
                description=_ICD_CODES["ketoacidosis"][1],
                onset_date=onset,
                is_active=False,
                resolved_date=onset + timedelta(days=7),
            ))

        # Hyperlipidemia
        if random.random() < 0.65:
            diagnoses.append(ICD10Diagnosis(
                icd10_code=_ICD_CODES["hyperlipidemia"][0],
                description=_ICD_CODES["hyperlipidemia"][1],
                onset_date=_random_date_before(enrollment_date, 365 * 6),
                is_active=True,
            ))

        return diagnoses

    def _generate_medications(
        self, enrollment_date: date, diagnoses: list[ICD10Diagnosis]
    ) -> list[RxNormMedication]:
        meds = []
        dx_codes = {dx.icd10_code for dx in diagnoses}

        # Metformin if T2DM
        if any(c.startswith("E11") for c in dx_codes):
            start = _random_date_before(enrollment_date, 365 * 5)
            meds.append(RxNormMedication(
                drug_name="metformin",
                rxnorm_code="860974",
                dose_mg=1000,
                route="oral",
                indication="type 2 diabetes mellitus",
                start_date=start,
            ))

        # Aspirin for MI
        if random.random() < N2C2_CRITERIA_PREVALENCE["ASP_FOR_MI"]:
            start = _random_date_before(enrollment_date, 365 * 3)
            meds.append(RxNormMedication(
                drug_name="aspirin",
                rxnorm_code="1191",
                dose_mg=81,
                route="oral",
                indication="MI prevention",
                start_date=start,
            ))

        # Statin if CAD or hyperlipidemia
        if any(c.startswith(("I25", "E78")) for c in dx_codes):
            start = _random_date_before(enrollment_date, 365 * 4)
            meds.append(RxNormMedication(
                drug_name="atorvastatin",
                rxnorm_code="83367",
                dose_mg=40,
                route="oral",
                indication="hyperlipidemia",
                start_date=start,
            ))

        # ACE inhibitor if HTN
        if any(c == "I10" for c in dx_codes):
            start = _random_date_before(enrollment_date, 365 * 4)
            meds.append(RxNormMedication(
                drug_name="lisinopril",
                rxnorm_code="29046",
                dose_mg=10,
                route="oral",
                indication="hypertension",
                start_date=start,
            ))

        return meds

    def _generate_labs(
        self, enrollment_date: date, diagnoses: list[ICD10Diagnosis]
    ) -> list[LabResult]:
        labs = []
        dx_codes = {dx.icd10_code for dx in diagnoses}

        # HbA1c — n2c2 prevalence-based
        has_t2dm = any(c.startswith("E11") for c in dx_codes)
        n_hba1c = random.randint(2, 4) if has_t2dm else random.randint(0, 1)

        for i in range(n_hba1c):
            days_back = random.randint(10, 180 * (i + 1))
            if random.random() < N2C2_CRITERIA_PREVALENCE["HBA1C"]:
                value = random.gauss(7.5, 0.8)
                value = max(6.5, min(9.5, value))
            else:
                value = random.choice([
                    random.gauss(5.8, 0.3),
                    random.gauss(11.0, 1.0),
                ])
            labs.append(LabResult(
                lab_name="HbA1c",
                loinc_code="4548-4",
                value=round(value, 1),
                unit="%",
                reference_low=4.0,
                reference_high=5.6,
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=days_back),
                    datetime.min.time(),
                ),
            ))

        # Creatinine
        has_ckd = any(c.startswith("N18") for c in dx_codes)
        if random.random() < N2C2_CRITERIA_PREVALENCE["CREATININE"] or has_ckd:
            value = random.gauss(1.8, 0.4) if has_ckd else random.gauss(1.4, 0.3)
        else:
            value = random.gauss(0.95, 0.2)

        labs.append(LabResult(
            lab_name="Creatinine",
            loinc_code="2160-0",
            value=round(max(0.4, value), 2),
            unit="mg/dL",
            reference_low=0.6,
            reference_high=1.2,
            result_datetime=datetime.combine(
                enrollment_date - timedelta(days=random.randint(7, 90)),
                datetime.min.time(),
            ),
        ))

        # Fasting glucose
        if has_t2dm:
            labs.append(LabResult(
                lab_name="Fasting Glucose",
                loinc_code="1558-6",
                value=round(random.gauss(145, 35), 0),
                unit="mg/dL",
                reference_low=70,
                reference_high=100,
                result_datetime=datetime.combine(
                    enrollment_date - timedelta(days=random.randint(7, 60)),
                    datetime.min.time(),
                ),
            ))

        return labs

    def _generate_vitals(self, enrollment_date: date) -> list[VitalSign]:
        bmi = random.gauss(30, 6)
        weight = bmi * (1.75 ** 2)
        return [
            VitalSign(
                vital_name="bmi",
                value=round(max(18, min(50, bmi)), 1),
                unit="kg/m2",
                measured_datetime=datetime.combine(
                    enrollment_date - timedelta(days=random.randint(1, 30)),
                    datetime.min.time(),
                ),
            ),
            VitalSign(
                vital_name="weight_kg",
                value=round(max(50, weight), 1),
                unit="kg",
                measured_datetime=datetime.combine(
                    enrollment_date - timedelta(days=random.randint(1, 30)),
                    datetime.min.time(),
                ),
            ),
        ]


# ── Demo patients (hardcoded for consistent Streamlit Cloud demo) ─────────────

def generate_demo_patients() -> list[PatientProfile]:
    """Generate 5 representative demo patients with fixed seeds."""
    patients = []
    configs = [
        (42, "Well-controlled T2DM, aspirin, recent MI"),
        (7, "Poorly-controlled T2DM, no aspirin, old MI"),
        (99, "T2DM with CKD, high creatinine"),
        (15, "T2DM, alcohol abuse, dietary supplements"),
        (23, "Mild T2DM, no complications"),
    ]
    enrollment = date(2024, 6, 1)
    for seed, _ in configs:
        gen = SyntheticPatientGenerator(seed=seed)
        p = gen.generate(1, enrollment_date=enrollment)[0]
        patients.append(p)
    return patients
