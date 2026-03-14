"""Typed eligibility criterion schema — discriminated union hierarchy."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class CriterionType(str, Enum):
    DEMOGRAPHIC = "DEMOGRAPHIC"
    LAB_VALUE = "LAB_VALUE"
    DIAGNOSIS = "DIAGNOSIS"
    MEDICATION = "MEDICATION"
    PROCEDURE = "PROCEDURE"
    BEHAVIORAL = "BEHAVIORAL"
    FUNCTIONAL = "FUNCTIONAL"
    COMPOSITE = "COMPOSITE"
    UNKNOWN = "UNKNOWN"


class TemporalConstraint(BaseModel):
    operator: Literal["WITHIN", "PRIOR_TO", "AFTER", "AT_TIME_OF", "EVER"]
    value: float | None = None
    unit: Literal["days", "months", "years"] | None = None
    raw_text: str = ""

    @property
    def as_timedelta(self) -> timedelta | None:
        if self.operator == "EVER" or self.value is None or self.unit is None:
            return None
        days_map = {"days": 1, "months": 30, "years": 365}
        return timedelta(days=self.value * days_map[self.unit])


class NumericThreshold(BaseModel):
    operator: Literal[">", ">=", "<", "<=", "=", "BETWEEN"]
    value: float
    value_high: float | None = None  # for BETWEEN
    unit: str = ""

    def evaluate(self, measured: float) -> bool:
        ops = {
            ">": measured > self.value,
            ">=": measured >= self.value,
            "<": measured < self.value,
            "<=": measured <= self.value,
            "=": abs(measured - self.value) < 1e-9,
            "BETWEEN": self.value <= measured <= (self.value_high or self.value),
        }
        return ops[self.operator]


# ── Leaf criterion types ────────────────────────────────────────────────────


class DemographicCriterion(BaseModel):
    type: Literal[CriterionType.DEMOGRAPHIC] = CriterionType.DEMOGRAPHIC
    attribute: Literal["age", "sex", "race", "ethnicity", "bmi", "weight"]
    threshold: NumericThreshold | None = None
    text_value: str | None = None  # for categorical attributes like sex
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class LabValueCriterion(BaseModel):
    type: Literal[CriterionType.LAB_VALUE] = CriterionType.LAB_VALUE
    lab_name: str
    loinc_code: str | None = None
    threshold: NumericThreshold
    temporal: TemporalConstraint | None = None
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class DiagnosisCriterion(BaseModel):
    type: Literal[CriterionType.DIAGNOSIS] = CriterionType.DIAGNOSIS
    condition_name: str
    icd10_code: str | None = None
    temporal: TemporalConstraint | None = None
    required_absent: bool = False  # True = exclusion criterion (no history of X)
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class MedicationCriterion(BaseModel):
    type: Literal[CriterionType.MEDICATION] = CriterionType.MEDICATION
    drug_name: str
    rxnorm_code: str | None = None
    dose_mg: float | None = None
    indication: str | None = None
    temporal: TemporalConstraint | None = None
    required_absent: bool = False
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class ProcedureCriterion(BaseModel):
    type: Literal[CriterionType.PROCEDURE] = CriterionType.PROCEDURE
    procedure_name: str
    cpt_code: str | None = None
    temporal: TemporalConstraint | None = None
    required_absent: bool = False
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class BehavioralCriterion(BaseModel):
    type: Literal[CriterionType.BEHAVIORAL] = CriterionType.BEHAVIORAL
    behavior: Literal[
        "DRUG_ABUSE", "ALCOHOL_ABUSE", "SMOKING", "DIETARY_SUPPLEMENT", "EXERCISE", "OTHER"
    ]
    details: str = ""
    temporal: TemporalConstraint | None = None
    required_absent: bool = False
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class FunctionalCriterion(BaseModel):
    type: Literal[CriterionType.FUNCTIONAL] = CriterionType.FUNCTIONAL
    capability: str  # e.g., "makes own decisions", "ambulatory"
    scale: str | None = None  # e.g., "ECOG", "KPS"
    threshold: NumericThreshold | None = None
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


class UnknownCriterion(BaseModel):
    type: Literal[CriterionType.UNKNOWN] = CriterionType.UNKNOWN
    raw_text: str = ""
    ambiguous: bool = True
    ambiguity_note: str = "Could not parse criterion type"


# ── Composite ────────────────────────────────────────────────────────────────


class CompositeCriterion(BaseModel):
    type: Literal[CriterionType.COMPOSITE] = CriterionType.COMPOSITE
    operator: Literal["AND", "OR", "NOT", "AT_LEAST_N"]
    n: int | None = None  # for AT_LEAST_N
    children: list["ParsedCriterion"] = Field(default_factory=list)
    raw_text: str = ""
    ambiguous: bool = False
    ambiguity_note: str = ""


# ── Top-level discriminated union ────────────────────────────────────────────

ParsedCriterion = Annotated[
    Union[
        LabValueCriterion,
        DiagnosisCriterion,
        MedicationCriterion,
        BehavioralCriterion,
        DemographicCriterion,
        FunctionalCriterion,
        ProcedureCriterion,
        CompositeCriterion,
        UnknownCriterion,
    ],
    Field(discriminator="type"),
]

CompositeCriterion.model_rebuild()


class EligibilityCriteriaSet(BaseModel):
    nct_id: str
    raw_text: str
    inclusion: list[ParsedCriterion] = Field(default_factory=list)
    exclusion: list[ParsedCriterion] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
    parser_version: str = "0.1.0"
