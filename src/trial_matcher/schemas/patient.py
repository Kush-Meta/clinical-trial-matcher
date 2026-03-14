"""PatientProfile schema — structured patient chart representation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ICD10Diagnosis(BaseModel):
    icd10_code: str
    description: str = ""
    onset_date: date | None = None
    resolved_date: date | None = None
    is_active: bool = True
    source: Literal["structured", "note_nlp"] = "structured"
    confidence: float = 1.0

    def active_at(self, ref_date: date) -> bool:
        if self.onset_date and self.onset_date > ref_date:
            return False
        if self.resolved_date and self.resolved_date < ref_date:
            return False
        return True


class RxNormMedication(BaseModel):
    drug_name: str
    rxnorm_code: str | None = None
    dose_mg: float | None = None
    dose_unit: str = "mg"
    route: str = ""
    indication: str = ""
    start_date: date | None = None
    stop_date: date | None = None
    source: Literal["structured", "note_nlp"] = "structured"
    confidence: float = 1.0

    def active_at(self, ref_date: date) -> bool:
        if self.start_date and self.start_date > ref_date:
            return False
        if self.stop_date and self.stop_date < ref_date:
            return False
        return True


class LabResult(BaseModel):
    lab_name: str
    loinc_code: str | None = None
    value: float
    unit: str = ""
    reference_low: float | None = None
    reference_high: float | None = None
    result_datetime: datetime
    source: Literal["structured", "note_nlp"] = "structured"
    confidence: float = 1.0
    is_abnormal: bool = False

    @property
    def result_date(self) -> date:
        return self.result_datetime.date()


class Procedure(BaseModel):
    procedure_name: str
    cpt_code: str | None = None
    procedure_date: date | None = None
    source: Literal["structured", "note_nlp"] = "structured"
    confidence: float = 1.0


class VitalSign(BaseModel):
    vital_name: str  # e.g., "blood_pressure_systolic", "bmi", "weight_kg"
    value: float
    unit: str = ""
    measured_datetime: datetime
    source: Literal["structured", "note_nlp"] = "structured"


class ExtractedEntity(BaseModel):
    text: str
    entity_type: str  # e.g., "DISEASE", "DRUG", "LAB"
    is_negated: bool = False
    is_uncertain: bool = False
    section: str = ""
    normalized_code: str | None = None
    normalized_date: date | None = None
    confidence: float = 1.0


class ClinicalNote(BaseModel):
    note_id: str
    note_type: str  # e.g., "discharge_summary", "progress_note"
    note_date: date | None = None
    text: str = ""
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)


class PatientProfile(BaseModel):
    patient_id: str
    source: Literal["mimic_iv", "synthetic", "n2c2", "uploaded"]
    age: int | None = None
    sex: Literal["M", "F", "Unknown"] = "Unknown"
    enrollment_date: date

    diagnoses: list[ICD10Diagnosis] = Field(default_factory=list)
    medications: list[RxNormMedication] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    vitals: list[VitalSign] = Field(default_factory=list)
    notes: list[ClinicalNote] = Field(default_factory=list)

    # Behavioral flags (None = unknown)
    has_drug_abuse: bool | None = None
    has_alcohol_abuse: bool | None = None
    has_smoking_history: bool | None = None
    dietary_supplements: list[str] = Field(default_factory=list)

    data_completeness_score: float = 1.0

    def get_labs_by_name(self, name: str, loinc: str | None = None) -> list[LabResult]:
        results = []
        name_lower = name.lower()
        for lab in self.labs:
            if loinc and lab.loinc_code == loinc:
                results.append(lab)
            elif name_lower in lab.lab_name.lower() or lab.lab_name.lower() in name_lower:
                results.append(lab)
        return results

    def get_active_medications(self, ref_date: date) -> list[RxNormMedication]:
        return [m for m in self.medications if m.active_at(ref_date)]

    def get_active_diagnoses(self, ref_date: date) -> list[ICD10Diagnosis]:
        return [d for d in self.diagnoses if d.active_at(ref_date)]
