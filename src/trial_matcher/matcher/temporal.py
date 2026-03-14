"""Temporal window evaluation and date arithmetic."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from trial_matcher.schemas.criteria import TemporalConstraint

logger = logging.getLogger(__name__)

# Uncertainty band for year-only dates (±180 days from July 1 midpoint)
YEAR_ONLY_UNCERTAINTY_DAYS = 180

# Confidence degradation near boundary
BOUNDARY_PENALTY_PER_PERCENT = 0.0015  # 15% penalty at boundary = 1.0 - 0.0015*100


def evaluate_in_window(
    event_date: date | None,
    enrollment_date: date,
    constraint: TemporalConstraint,
    date_precision: str = "exact",  # "exact", "month", "year"
) -> tuple[bool | None, float]:
    """
    Evaluate whether an event falls within a temporal constraint window.

    Returns:
        (in_window, confidence)
        - in_window=None → ABSTAIN (insufficient info)
        - in_window=True/False → definite answer
        - confidence degrades near window boundaries
    """
    if event_date is None:
        return None, 0.0

    if constraint.operator == "EVER":
        # Any occurrence counts
        return True, 1.0

    if constraint.operator == "AT_TIME_OF":
        # Within ±30 days of enrollment
        days_off = abs((event_date - enrollment_date).days)
        in_window = days_off <= 30
        confidence = max(0.0, 1.0 - days_off / 60)
        return in_window, confidence

    window_delta = constraint.as_timedelta

    if window_delta is None:
        # No temporal constraint to evaluate
        return True, 1.0

    if date_precision == "year":
        # Use July 1 as midpoint with uncertainty band
        event_mid = date(event_date.year, 7, 1)
        uncertainty = timedelta(days=YEAR_ONLY_UNCERTAINTY_DAYS)

        if constraint.operator == "WITHIN":
            deadline = enrollment_date - window_delta
            # Definitely in window
            if event_mid - uncertainty >= deadline:
                return True, _boundary_confidence(event_mid, deadline, window_delta)
            # Definitely out
            if event_mid + uncertainty < deadline:
                return False, 1.0
            # Ambiguous
            return None, 0.0

    # Standard date evaluation
    if constraint.operator == "WITHIN":
        deadline = enrollment_date - window_delta
        in_window = event_date >= deadline
        confidence = _boundary_confidence(event_date, deadline, window_delta)
        return in_window, confidence

    elif constraint.operator == "PRIOR_TO":
        in_window = event_date < enrollment_date - window_delta
        confidence = _boundary_confidence(event_date, enrollment_date, window_delta)
        return in_window, confidence

    elif constraint.operator == "AFTER":
        target = enrollment_date - window_delta
        in_window = event_date > target
        confidence = _boundary_confidence(event_date, target, window_delta)
        return in_window, confidence

    return None, 0.0


def _boundary_confidence(
    event_date: date, boundary_date: date, window: timedelta
) -> float:
    """
    Confidence degrades as event_date approaches the boundary.
    Full confidence if event is >10% of window from boundary.
    """
    if window.days == 0:
        return 1.0

    days_from_boundary = abs((event_date - boundary_date).days)
    margin_fraction = days_from_boundary / window.days

    if margin_fraction >= 0.10:
        return 1.0
    # Linear degradation in the 0–10% zone
    return 0.5 + 0.5 * (margin_fraction / 0.10)


def get_most_recent(
    items: list,
    date_getter: callable,
    before: date | None = None,
) -> object | None:
    """Get most recent item, optionally filtered to those before a cutoff."""
    filtered = [item for item in items if date_getter(item) is not None]
    if before:
        filtered = [item for item in filtered if date_getter(item) <= before]
    if not filtered:
        return None
    return max(filtered, key=lambda x: date_getter(x))


def days_since(event_date: date, ref_date: date) -> int:
    return (ref_date - event_date).days
