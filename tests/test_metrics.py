"""Tests for binary classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cervical_biopsy_modeling.metrics import (
    BinaryMetrics,
    compute_binary_metrics,
)


def test_compute_binary_metrics_known_example() -> None:
    """All threshold-based counts must match by inspection."""
    outcome = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.4, 0.9])

    metrics = compute_binary_metrics(
        outcome,
        probabilities,
    )

    assert metrics.average_precision == pytest.approx(5 / 6)
    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.brier_score == pytest.approx(0.2175)
    assert metrics.sensitivity == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.positive_predictive_value == pytest.approx(0.5)
    assert metrics.f1_score == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.5)
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.threshold == pytest.approx(0.5)


def test_compute_binary_metrics_perfect_predictions() -> None:
    """Perfect ranking and classification must score one."""
    outcome = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_binary_metrics(
        outcome,
        probabilities,
    )

    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.brier_score == pytest.approx(0.025)
    assert metrics.sensitivity == pytest.approx(1.0)
    assert metrics.specificity == pytest.approx(1.0)
    assert metrics.positive_predictive_value == pytest.approx(1.0)
    assert metrics.f1_score == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.true_positives == 2


def test_compute_binary_metrics_custom_threshold() -> None:
    """Threshold-based metrics must respect the given cutoff."""
    outcome = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.6, 0.4, 0.8])

    metrics = compute_binary_metrics(
        outcome,
        probabilities,
        threshold=0.7,
    )

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 1
    assert metrics.sensitivity == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(1.0)
    assert metrics.positive_predictive_value == pytest.approx(1.0)
    assert metrics.f1_score == pytest.approx(2 / 3)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.threshold == pytest.approx(0.7)


def test_binary_metrics_serializes_to_dictionary() -> None:
    """Metric records must be ready for tabular output."""
    outcome = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_binary_metrics(
        outcome,
        probabilities,
    )
    serialized = metrics.to_dict()

    assert isinstance(metrics, BinaryMetrics)
    assert serialized["average_precision"] == pytest.approx(1.0)
    assert serialized["true_negatives"] == 2
    assert serialized["true_positives"] == 2
    assert serialized["threshold"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    (
        "outcome",
        "probabilities",
        "threshold",
        "expected_message",
    ),
    [
        (
            [0, 1],
            [0.1],
            0.5,
            "equal length",
        ),
        (
            [0, 1],
            [0.1, np.nan],
            0.5,
            "finite",
        ),
        (
            [0, 1],
            [-0.1, 0.9],
            0.5,
            "between 0 and 1",
        ),
        (
            [0, 0],
            [0.1, 0.2],
            0.5,
            "both binary classes",
        ),
        (
            [0, 1],
            [0.1, 0.9],
            1.1,
            "threshold",
        ),
    ],
)
def test_compute_binary_metrics_rejects_invalid_inputs(
    outcome: list[int],
    probabilities: list[float],
    threshold: float,
    expected_message: str,
) -> None:
    """Invalid metric inputs must fail explicitly."""
    with pytest.raises(ValueError, match=expected_message):
        compute_binary_metrics(
            outcome,
            probabilities,
            threshold=threshold,
        )
