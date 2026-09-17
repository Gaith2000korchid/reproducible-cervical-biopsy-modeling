"""Binary classification metrics for outer-fold evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class BinaryMetrics:
    """Threshold-free and threshold-based binary metrics."""

    average_precision: float
    roc_auc: float
    brier_score: float
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    f1_score: float
    balanced_accuracy: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    threshold: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialization-ready metric dictionary."""
        return asdict(self)


def _validate_binary_inputs(
    outcome: Any,
    probabilities: Any,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize binary metric inputs."""
    outcome_array = np.asarray(outcome)
    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if outcome_array.ndim != 1:
        raise ValueError("Outcome must be one-dimensional.")

    if probability_array.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional.")

    if len(outcome_array) == 0:
        raise ValueError("At least one observation is required.")

    if len(outcome_array) != len(probability_array):
        raise ValueError("Outcome and probabilities must have equal length.")

    if not np.array_equal(
        np.unique(outcome_array),
        np.array([0, 1]),
    ):
        raise ValueError("Outcome must contain both binary classes 0 and 1.")

    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must be finite.")

    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError("Probabilities must lie between 0 and 1.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must lie between 0 and 1.")

    return outcome_array.astype(int), probability_array


def compute_binary_metrics(
    outcome: Any,
    probabilities: Any,
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> BinaryMetrics:
    """Compute prespecified metrics from positive probabilities."""
    outcome_array, probability_array = _validate_binary_inputs(
        outcome,
        probabilities,
        threshold=threshold,
    )

    predictions = (probability_array >= threshold).astype(int)

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        outcome_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity_denominator = true_negatives + false_positives
    specificity = true_negatives / specificity_denominator

    return BinaryMetrics(
        average_precision=float(
            average_precision_score(
                outcome_array,
                probability_array,
            )
        ),
        roc_auc=float(
            roc_auc_score(
                outcome_array,
                probability_array,
            )
        ),
        brier_score=float(
            brier_score_loss(
                outcome_array,
                probability_array,
            )
        ),
        sensitivity=float(
            recall_score(
                outcome_array,
                predictions,
                zero_division=0,
            )
        ),
        specificity=float(specificity),
        positive_predictive_value=float(
            precision_score(
                outcome_array,
                predictions,
                zero_division=0,
            )
        ),
        f1_score=float(
            f1_score(
                outcome_array,
                predictions,
                zero_division=0,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(
                outcome_array,
                predictions,
            )
        ),
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
        threshold=float(threshold),
    )
