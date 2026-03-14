"""Evaluation metrics: P/R/F1, ECE, coverage-accuracy tradeoff."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CriterionMetrics:
    criterion: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    coverage: float  # fraction not abstained
    abstention_rate: float
    support: int


@dataclass
class EvaluationResult:
    criterion_metrics: list[CriterionMetrics] = field(default_factory=list)
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    ece: float = 0.0
    n_patients: int = 0
    n_pairs: int = 0
    overall_coverage: float = 0.0
    coverage_accuracy_curve: list[tuple[float, float]] = field(default_factory=list)


def compute_criterion_metrics(
    labels: list[int],           # 0, 1
    predictions: list[int],      # 0, 1, -1 (abstain)
    confidences: list[float],
) -> dict[str, float]:
    """Compute P/R/F1 for a single criterion, ignoring abstentions."""
    assert len(labels) == len(predictions) == len(confidences)

    # Filter out abstentions
    kept_idx = [i for i, p in enumerate(predictions) if p != -1]
    if not kept_idx:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
                "coverage": 0.0, "abstention_rate": 1.0}

    y_true = [labels[i] for i in kept_idx]
    y_pred = [predictions[i] for i in kept_idx]

    coverage = len(kept_idx) / len(labels)
    abstention_rate = 1.0 - coverage

    # TP, FP, FN
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "coverage": coverage,
        "abstention_rate": abstention_rate,
    }


def compute_ece(
    labels: list[int],
    predictions: list[int],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b|/N) * |acc(B_b) - conf(B_b)|
    """
    # Filter abstentions
    kept = [(l, p, c) for l, p, c in zip(labels, predictions, confidences) if p != -1]
    if not kept:
        return 0.0

    labels_k, preds_k, confs_k = zip(*kept)
    n = len(labels_k)

    # Bin by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        bin_idx = [
            j for j, c in enumerate(confs_k) if low <= c < high
        ]
        if not bin_idx:
            continue

        bin_conf = np.mean([confs_k[j] for j in bin_idx])
        bin_acc = np.mean([
            1.0 if labels_k[j] == preds_k[j] else 0.0
            for j in bin_idx
        ])
        ece += (len(bin_idx) / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_coverage_accuracy_curve(
    labels: list[int],
    confidences: list[float],
    predictions_at_threshold: callable,
    thresholds: list[float] | None = None,
) -> list[tuple[float, float, float]]:
    """
    Coverage-accuracy tradeoff curve.

    For each abstention threshold t:
      - Keep examples with confidence >= t
      - Compute (threshold, coverage, accuracy)

    Returns list of (threshold, coverage, accuracy) tuples.
    """
    if thresholds is None:
        thresholds = [i / 20 for i in range(21)]  # 0.0 to 1.0 in 0.05 steps

    curve = []
    for t in thresholds:
        preds = predictions_at_threshold(t)
        kept = [(l, p) for l, p, c in zip(labels, preds, confidences) if p != -1]
        if not kept:
            curve.append((t, 0.0, 0.0))
            continue

        coverage = len(kept) / len(labels)
        labels_k, preds_k = zip(*kept)
        accuracy = sum(1 for l, p in zip(labels_k, preds_k) if l == p) / len(kept)
        curve.append((t, coverage, accuracy))

    return curve


def evaluate_batch(
    all_labels: dict[str, list[int]],        # criterion → labels
    all_predictions: dict[str, list[int]],   # criterion → predictions
    all_confidences: dict[str, list[float]], # criterion → confidences
) -> EvaluationResult:
    """Compute full evaluation across all criteria."""
    criterion_metrics = []
    macro_p, macro_r, macro_f = [], [], []

    for criterion in all_labels:
        labels = all_labels[criterion]
        preds = all_predictions.get(criterion, [-1] * len(labels))
        confs = all_confidences.get(criterion, [0.0] * len(labels))

        m = compute_criterion_metrics(labels, preds, confs)
        criterion_metrics.append(CriterionMetrics(
            criterion=criterion,
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            accuracy=m["accuracy"],
            coverage=m["coverage"],
            abstention_rate=m["abstention_rate"],
            support=len(labels),
        ))
        if m["coverage"] > 0:
            macro_p.append(m["precision"])
            macro_r.append(m["recall"])
            macro_f.append(m["f1"])

    # Flatten for ECE
    flat_labels = [l for labels in all_labels.values() for l in labels]
    flat_preds = [p for preds in all_predictions.values() for p in preds]
    flat_confs = [c for confs in all_confidences.values() for c in confs]

    ece = compute_ece(flat_labels, flat_preds, flat_confs)
    n_kept = sum(1 for p in flat_preds if p != -1)
    coverage = n_kept / len(flat_preds) if flat_preds else 0.0

    return EvaluationResult(
        criterion_metrics=criterion_metrics,
        macro_f1=float(np.mean(macro_f)) if macro_f else 0.0,
        macro_precision=float(np.mean(macro_p)) if macro_p else 0.0,
        macro_recall=float(np.mean(macro_r)) if macro_r else 0.0,
        ece=ece,
        n_patients=len(next(iter(all_labels.values()), [])),
        n_pairs=len(flat_labels),
        overall_coverage=coverage,
    )


def generate_reliability_diagram_data(
    labels: list[int],
    predictions: list[int],
    confidences: list[float],
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Return data for a reliability (calibration) diagram."""
    kept = [(l, p, c) for l, p, c in zip(labels, predictions, confidences) if p != -1]
    if not kept:
        return {"bin_centers": [], "mean_confidence": [], "accuracy": [], "counts": []}

    labels_k, preds_k, confs_k = zip(*kept)
    n = len(labels_k)
    bins = np.linspace(0, 1, n_bins + 1)

    centers, mean_confs, accs, counts = [], [], [], []
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        bin_idx = [j for j, c in enumerate(confs_k) if low <= c < high]
        if not bin_idx:
            continue
        centers.append((low + high) / 2)
        mean_confs.append(float(np.mean([confs_k[j] for j in bin_idx])))
        accs.append(float(np.mean([
            1.0 if labels_k[j] == preds_k[j] else 0.0
            for j in bin_idx
        ])))
        counts.append(len(bin_idx))

    return {
        "bin_centers": centers,
        "mean_confidence": mean_confs,
        "accuracy": accs,
        "counts": counts,
    }
