"""Tests for boolean criteria parser."""

from __future__ import annotations

import pytest

from trial_matcher.criteria_parser.boolean_parser import (
    detect_boolean_structure,
    extract_at_least_n,
    parse_composite,
    split_sub_items,
)


class TestDetectBooleanStructure:
    def test_at_least_n_detected(self) -> None:
        text = "Patient must have at least 2 of the following conditions"
        assert detect_boolean_structure(text) == "AT_LEAST_N"

    def test_any_of_detected(self) -> None:
        text = "Any one of the following: diabetes, hypertension, CAD"
        assert detect_boolean_structure(text) == "OR"

    def test_all_of_detected(self) -> None:
        text = "All of the following must be present: age > 18, diabetes, English"
        assert detect_boolean_structure(text) == "AND"

    def test_simple_no_structure(self) -> None:
        text = "HbA1c between 6.5% and 9.5% within 3 months"
        assert detect_boolean_structure(text) is None

    def test_two_or_more(self) -> None:
        text = "Two or more of the following cardiovascular conditions"
        assert detect_boolean_structure(text) == "AT_LEAST_N"


class TestExtractAtLeastN:
    def test_numeric_digit(self) -> None:
        text = "at least 3 of the following"
        assert extract_at_least_n(text) == 3

    def test_word_two(self) -> None:
        text = "two or more of the following criteria"
        assert extract_at_least_n(text) == 2

    def test_fallback_2(self) -> None:
        text = "several of the following conditions"
        assert extract_at_least_n(text) == 2


class TestSplitSubItems:
    def test_numbered_list(self) -> None:
        text = "1. History of MI 2. Coronary bypass 3. PCI"
        items = split_sub_items(text)
        assert len(items) >= 3

    def test_or_separated(self) -> None:
        text = "diabetes or hypertension or CAD"
        items = split_sub_items(text)
        assert len(items) >= 2

    def test_single_item(self) -> None:
        text = "HbA1c > 7%"
        items = split_sub_items(text)
        assert items == ["HbA1c > 7%"]


class TestParseComposite:
    def test_at_least_2_composite(self) -> None:
        text = (
            "Patient must have at least 2 of the following: "
            "1. History of MI 2. CABG 3. PCI"
        )
        result = parse_composite(text)
        assert result is not None
        assert result.operator == "AT_LEAST_N"
        assert result.n == 2
        assert len(result.children) >= 2

    def test_simple_text_no_composite(self) -> None:
        text = "HbA1c between 6.5% and 9.5%"
        result = parse_composite(text)
        assert result is None
