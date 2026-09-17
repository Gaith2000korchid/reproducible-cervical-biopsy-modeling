"""Tests for paired grouped-bootstrap model comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cervical_biopsy_modeling.comparisons import (
    compute_repeat_differences,
    grouped_bootstrap_comparisons,
    oriented_difference,
    save_comparisons,
)
from cervical_biopsy_modeling.summary import (
    PERFORMANCE_METRICS,
)


def make_prediction_table() -> pd.DataFrame:
    """Create paired dummy and perfect model predictions."""
    outcome = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.repeat(np.arange(4), 2)

    perfect_probabilities = {
        1: np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95]),
        2: np.array([0.10, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.90]),
    }

    records = []

    for model in ["dummy", "perfect"]:
        for repeat in [1, 2]:
            probabilities = (
                np.full(8, 0.5) if model == "dummy" else perfect_probabilities[repeat]
            )

            for row_position in range(8):
                records.append(
                    {
                        "repeat": repeat,
                        "model": model,
                        "row_position": row_position,
                        "group": groups[row_position],
                        "observed_outcome": outcome[row_position],
                        "predicted_probability": probabilities[row_position],
                        "threshold": 0.5,
                    }
                )

    return pd.DataFrame.from_records(records)


def test_oriented_difference_handles_brier_direction() -> None:
    """Positive differences must always favor the candidate."""
    assert oriented_difference(
        0.8,
        0.5,
        metric="roc_auc",
    ) == pytest.approx(0.3)

    assert oriented_difference(
        0.1,
        0.2,
        metric="brier_score",
    ) == pytest.approx(0.1)


def test_repeat_differences_cover_every_metric() -> None:
    """One model pair must yield all prespecified metrics."""
    differences = compute_repeat_differences(
        make_prediction_table(),
        comparisons=(("perfect", "dummy"),),
    )

    assert len(differences) == len(PERFORMANCE_METRICS)
    assert set(differences["metric"]) == set(PERFORMANCE_METRICS)
    assert (differences["candidate_model"] == "perfect").all()
    assert (differences["reference_model"] == "dummy").all()

    ap = differences[differences["metric"] == "average_precision"].iloc[0]
    assert ap["repeat_mean_difference"] == pytest.approx(0.5)

    sensitivity = differences[differences["metric"] == "sensitivity"].iloc[0]
    assert sensitivity["repeat_mean_difference"] == pytest.approx(0.0)


def test_paired_bootstrap_is_reproducible() -> None:
    """The same seed must reproduce every paired interval."""
    predictions = make_prediction_table()

    first = grouped_bootstrap_comparisons(
        predictions,
        comparisons=(("perfect", "dummy"),),
        n_resamples=25,
        random_state=211,
    )
    second = grouped_bootstrap_comparisons(
        predictions,
        comparisons=(("perfect", "dummy"),),
        n_resamples=25,
        random_state=211,
    )

    assert_frame_equal(first, second)


def test_paired_intervals_detect_expected_advantages() -> None:
    """Perfect predictions must outperform the dummy baseline."""
    comparisons = grouped_bootstrap_comparisons(
        make_prediction_table(),
        comparisons=(("perfect", "dummy"),),
        n_resamples=25,
        random_state=223,
    )

    assert len(comparisons) == len(PERFORMANCE_METRICS)

    sensitivity = comparisons[comparisons["metric"] == "sensitivity"].iloc[0]
    assert sensitivity["bootstrap_lower"] == pytest.approx(0.0)
    assert sensitivity["bootstrap_upper"] == pytest.approx(0.0)
    assert not bool(sensitivity["interval_excludes_zero"])

    other_metrics = comparisons[comparisons["metric"] != "sensitivity"]
    assert (other_metrics["bootstrap_lower"] > 0.0).all()
    assert other_metrics["interval_excludes_zero"].all()


def test_comparison_rejects_unavailable_model() -> None:
    """Every requested model must exist in predictions."""
    with pytest.raises(
        ValueError,
        match="unavailable models",
    ):
        grouped_bootstrap_comparisons(
            make_prediction_table(),
            comparisons=(("missing", "dummy"),),
            n_resamples=5,
        )


def test_save_comparisons_writes_csv(
    tmp_path: Path,
) -> None:
    """Paired comparison output must be saved without index."""
    comparisons = grouped_bootstrap_comparisons(
        make_prediction_table(),
        comparisons=(("perfect", "dummy"),),
        n_resamples=10,
        random_state=227,
    )

    path = save_comparisons(
        comparisons,
        reports_directory=tmp_path,
    )

    assert path.exists()
    saved = pd.read_csv(path)
    assert len(saved) == len(PERFORMANCE_METRICS)
    assert "Unnamed: 0" not in saved.columns
