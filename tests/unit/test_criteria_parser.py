"""Tests for criteria schemas and deterministic parsing logic."""

from __future__ import annotations

import pytest

from trial_matcher.criteria_parser.extractor import CriteriaExtractor
from trial_matcher.criteria_parser.prompts import SEGMENTATION_HEADERS
from trial_matcher.schemas.criteria import (
    CompositeCriterion,
    CriterionType,
    DiagnosisCriterion,
    EligibilityCriteriaSet,
    LabValueCriterion,
    MedicationCriterion,
    NumericThreshold,
    TemporalConstraint,
)


class TestNumericThreshold:
    def test_greater_than(self) -> None:
        t = NumericThreshold(operator=">", value=7.0, unit="%")
        assert t.evaluate(7.5) is True
        assert t.evaluate(7.0) is False
        assert t.evaluate(6.5) is False

    def test_between(self) -> None:
        t = NumericThreshold(operator="BETWEEN", value=6.5, value_high=9.5, unit="%")
        assert t.evaluate(7.5) is True
        assert t.evaluate(6.5) is True
        assert t.evaluate(9.5) is True
        assert t.evaluate(5.0) is False
        assert t.evaluate(10.0) is False

    def test_less_than_equal(self) -> None:
        t = NumericThreshold(operator="<=", value=1.2, unit="mg/dL")
        assert t.evaluate(1.0) is True
        assert t.evaluate(1.2) is True
        assert t.evaluate(1.3) is False

    def test_equal(self) -> None:
        t = NumericThreshold(operator="=", value=81.0)
        assert t.evaluate(81.0) is True
        assert t.evaluate(82.0) is False


class TestTemporalConstraint:
    def test_ever_no_timedelta(self) -> None:
        tc = TemporalConstraint(operator="EVER")
        assert tc.as_timedelta is None

    def test_within_6_months_timedelta(self) -> None:
        from datetime import timedelta
        tc = TemporalConstraint(operator="WITHIN", value=6, unit="months")
        td = tc.as_timedelta
        assert td is not None
        assert td.days == 180

    def test_within_1_year_timedelta(self) -> None:
        from datetime import timedelta
        tc = TemporalConstraint(operator="WITHIN", value=1, unit="years")
        td = tc.as_timedelta
        assert td is not None
        assert td.days == 365


class TestSchemaValidation:
    def test_lab_value_criterion(self) -> None:
        c = LabValueCriterion(
            lab_name="HbA1c",
            threshold=NumericThreshold(operator="BETWEEN", value=6.5, value_high=9.5, unit="%"),
            temporal=TemporalConstraint(operator="WITHIN", value=3, unit="months"),
        )
        assert c.type == CriterionType.LAB_VALUE
        assert c.threshold.evaluate(7.5) is True

    def test_diagnosis_criterion(self) -> None:
        c = DiagnosisCriterion(
            condition_name="myocardial infarction",
            icd10_code="I21.9",
            temporal=TemporalConstraint(operator="WITHIN", value=6, unit="months"),
        )
        assert c.type == CriterionType.DIAGNOSIS
        assert c.required_absent is False

    def test_composite_criterion(self) -> None:
        children = [
            DiagnosisCriterion(
                condition_name="MI",
                temporal=TemporalConstraint(operator="EVER"),
            ),
            DiagnosisCriterion(
                condition_name="CAD",
                temporal=TemporalConstraint(operator="EVER"),
            ),
        ]
        c = CompositeCriterion(
            operator="AT_LEAST_N",
            n=2,
            children=children,
        )
        assert c.type == CriterionType.COMPOSITE
        assert len(c.children) == 2

    def test_eligibility_criteria_set_roundtrip(self) -> None:
        cs = EligibilityCriteriaSet(
            nct_id="NCT12345678",
            raw_text="Inclusion Criteria:\n- HbA1c > 7%\nExclusion Criteria:\n- Drug abuse",
            inclusion=[
                LabValueCriterion(
                    lab_name="HbA1c",
                    threshold=NumericThreshold(operator=">", value=7.0),
                )
            ],
            exclusion=[
                DiagnosisCriterion(condition_name="drug abuse", required_absent=True)
            ],
        )
        # Test JSON round-trip
        json_str = cs.model_dump_json()
        cs2 = EligibilityCriteriaSet.model_validate_json(json_str)
        assert cs2.nct_id == "NCT12345678"
        assert len(cs2.inclusion) == 1
        assert len(cs2.exclusion) == 1


class TestSegmentation:
    """Test deterministic section splitting (no LLM required)."""

    def _make_extractor(self) -> CriteriaExtractor:
        from unittest.mock import MagicMock
        settings = MagicMock()
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_model = "llama3.1:8b"
        return CriteriaExtractor(settings)

    def test_splits_inclusion_exclusion(self) -> None:
        text = """Inclusion Criteria:
- Age >= 18 years
- HbA1c > 6.5%

Exclusion Criteria:
- History of drug abuse
- Renal failure"""
        extractor = self._make_extractor()
        incl, excl = extractor._split_sections(text)
        assert len(incl) == 2
        assert len(excl) == 2

    def test_handles_colon_headers(self) -> None:
        text = """Inclusion Criteria:
1. Type 2 diabetes diagnosis

Exclusion Criteria:
1. Active malignancy"""
        extractor = self._make_extractor()
        incl, excl = extractor._split_sections(text)
        assert "Type 2 diabetes" in incl[0]
        assert "Active malignancy" in excl[0]

    def test_cleans_bullet_points(self) -> None:
        text = "Inclusion Criteria:\n• HbA1c > 7%\n* Creatinine < 1.5 mg/dL"
        extractor = self._make_extractor()
        incl, excl = extractor._split_sections(text)
        assert all("•" not in line and "*" not in line for line in incl)

    def test_short_lines_filtered(self) -> None:
        text = "Inclusion Criteria:\nYes\nAge >= 18 years old and able to consent"
        extractor = self._make_extractor()
        incl, excl = extractor._split_sections(text)
        # "Yes" (3 chars) should be filtered out
        assert all(len(line) >= 5 for line in incl)
