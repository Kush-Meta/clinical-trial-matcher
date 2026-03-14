"""Tests for confidence penalty model."""

from __future__ import annotations

import pytest

from trial_matcher.matcher.confidence import (
    PENALTY_NOTE_NLP_SOURCE,
    PENALTY_OLD_LAB,
    PENALTY_SINGLE_DATA_POINT,
    ConfidenceModel,
    apply_data_quality_penalties,
    compute_composite_confidence,
)
from trial_matcher.schemas.match import MatchStatus


class TestConfidenceModel:
    def test_no_penalties(self) -> None:
        model = ConfidenceModel()
        assert model.score == 1.0

    def test_single_penalty(self) -> None:
        model = ConfidenceModel()
        model.penalize("test", 0.2)
        assert abs(model.score - 0.8) < 1e-9

    def test_multiple_penalties(self) -> None:
        model = ConfidenceModel()
        model.penalize("a", 0.1)
        model.penalize("b", 0.2)
        assert abs(model.score - 0.7) < 1e-9

    def test_score_floor_at_zero(self) -> None:
        model = ConfidenceModel()
        model.penalize("a", 0.6)
        model.penalize("b", 0.6)
        assert model.score == 0.0

    def test_should_abstain_below_threshold(self) -> None:
        model = ConfidenceModel()
        model.penalize("missing_data", 0.7)
        should_abstain, reason = model.should_abstain(0.35)
        assert should_abstain is True

    def test_should_not_abstain_above_threshold(self) -> None:
        model = ConfidenceModel()
        model.penalize("minor", 0.1)
        should_abstain, reason = model.should_abstain(0.35)
        assert should_abstain is False
        assert reason is None


class TestApplyDataQualityPenalties:
    def test_note_nlp_penalty(self) -> None:
        model = ConfidenceModel()
        apply_data_quality_penalties(model, source="note_nlp")
        assert model.score < 1.0

    def test_structured_no_penalty(self) -> None:
        model = ConfidenceModel()
        apply_data_quality_penalties(model, source="structured", n_results=3)
        assert model.score == 1.0

    def test_old_lab_penalty(self) -> None:
        model = ConfidenceModel()
        apply_data_quality_penalties(model, days_old=100)
        assert model.score < 1.0

    def test_single_data_point_penalty(self) -> None:
        model = ConfidenceModel()
        apply_data_quality_penalties(model, n_results=1)
        assert model.score == 1.0 - PENALTY_SINGLE_DATA_POINT


class TestCompositeConfidence:
    def test_and_returns_min(self) -> None:
        conf = compute_composite_confidence(
            "AND",
            [0.9, 0.6, 0.8],
            [MatchStatus.SATISFIED] * 3,
        )
        assert conf == 0.6

    def test_or_returns_max(self) -> None:
        conf = compute_composite_confidence(
            "OR",
            [0.4, 0.9, 0.5],
            [MatchStatus.SATISFIED] * 3,
        )
        assert conf == 0.9

    def test_at_least_n_mean_of_top_n(self) -> None:
        conf = compute_composite_confidence(
            "AT_LEAST_N",
            [0.9, 0.7, 0.5, 0.3],
            [MatchStatus.SATISFIED, MatchStatus.SATISFIED, MatchStatus.NOT_SATISFIED, MatchStatus.SATISFIED],
            n=2,
        )
        # Top-2 satisfied: 0.9, 0.7 → mean = 0.8
        assert abs(conf - 0.8) < 1e-6

    def test_empty_returns_zero(self) -> None:
        conf = compute_composite_confidence("AND", [], [])
        assert conf == 0.0
