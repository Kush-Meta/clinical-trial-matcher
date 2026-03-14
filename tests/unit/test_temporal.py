"""Tests for temporal window evaluation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from trial_matcher.matcher.temporal import evaluate_in_window, days_since
from trial_matcher.schemas.criteria import TemporalConstraint


@pytest.fixture
def enrollment_date() -> date:
    return date(2024, 6, 1)


def make_constraint(operator: str, value: float | None = None, unit: str | None = None) -> TemporalConstraint:
    return TemporalConstraint(operator=operator, value=value, unit=unit, raw_text="test")


class TestEvaluateInWindow:
    def test_ever_always_true(self, enrollment_date: date) -> None:
        constraint = make_constraint("EVER")
        event = date(2020, 1, 1)
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint)
        assert in_window is True
        assert conf == 1.0

    def test_none_event_date(self, enrollment_date: date) -> None:
        constraint = make_constraint("WITHIN", 6, "months")
        in_window, conf = evaluate_in_window(None, enrollment_date, constraint)
        assert in_window is None
        assert conf == 0.0

    def test_within_6_months_satisfied(self, enrollment_date: date) -> None:
        constraint = make_constraint("WITHIN", 6, "months")
        event = enrollment_date - timedelta(days=90)  # 3 months ago
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint)
        assert in_window is True
        assert conf > 0.5

    def test_within_6_months_not_satisfied(self, enrollment_date: date) -> None:
        constraint = make_constraint("WITHIN", 6, "months")
        event = enrollment_date - timedelta(days=200)  # ~6.7 months ago
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint)
        assert in_window is False

    def test_within_6_months_boundary_confidence(self, enrollment_date: date) -> None:
        constraint = make_constraint("WITHIN", 6, "months")
        # Exactly at boundary (180 days = 6 months)
        event = enrollment_date - timedelta(days=179)
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint)
        assert in_window is True
        assert conf < 1.0  # degraded near boundary

    def test_at_time_of_within_margin(self, enrollment_date: date) -> None:
        constraint = make_constraint("AT_TIME_OF")
        event = enrollment_date - timedelta(days=15)  # within ±30 days
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint)
        assert in_window is True

    def test_at_time_of_outside_margin(self, enrollment_date: date) -> None:
        constraint = make_constraint("AT_TIME_OF")
        event = enrollment_date - timedelta(days=60)  # outside ±30 days
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint)
        assert in_window is False

    def test_year_only_ambiguous(self, enrollment_date: date) -> None:
        constraint = make_constraint("WITHIN", 3, "months")
        # July 1 of previous year ± 180 days overlaps 3-month window — ambiguous
        event = date(2024, 1, 1)  # year-only precision
        in_window, conf = evaluate_in_window(event, enrollment_date, constraint, date_precision="year")
        # Result can be None (ambiguous) or False — both acceptable
        assert in_window in (None, False)


class TestDaysSince:
    def test_basic(self) -> None:
        ref = date(2024, 6, 1)
        event = date(2024, 3, 1)  # ~92 days before
        assert days_since(event, ref) == 92

    def test_same_day(self) -> None:
        d = date(2024, 6, 1)
        assert days_since(d, d) == 0
