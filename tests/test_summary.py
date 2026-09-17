"""Tests for repeated-CV summaries and grouped bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cervical_biopsy_modeling.summary import (
    PERFORMANCE_METRICS,
    build_summary_results,
    compute_repeat_metrics,
    grouped_bootstrap_intervals,
    save_summary_results,
    summarize_repetitions,
    validate_prediction_table,
)


def make_prediction_table() -> pd.DataFrame:
    """Create aligned predictions for two models and repeats."""
    outcome = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.repeat(np.arange(4), 2)

    perfect_probabilities = {
        1: np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95]),
        2: np.array([0.10, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.90]),
    }

    records = []

    for model in ["dummy", "perfect"]:
        for repeat in [1, 2]:
            if model == "dummy":
                probabilities = np.full(8, 0.5)
            else:
                probabilities = perfect_probabilities[repeat]

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


def test_repeat_metrics_pool_complete_repetitions() -> None:
    """Each model and repeat must produce one pooled estimate."""
    repeat_metrics = compute_repeat_metrics(make_prediction_table())

    assert len(repeat_metrics) == 4
    assert (repeat_metrics["observations"] == 8).all()
    assert (repeat_metrics["positives"] == 4).all()

    dummy = repeat_metrics[repeat_metrics["model"] == "dummy"]
    perfect = repeat_metrics[repeat_metrics["model"] == "perfect"]

    assert np.allclose(dummy["average_precision"], 0.5)
    assert np.allclose(dummy["roc_auc"], 0.5)
    assert np.allclose(dummy["brier_score"], 0.25)
    assert np.allclose(dummy["sensitivity"], 1.0)
    assert np.allclose(dummy["specificity"], 0.0)
    assert np.allclose(
        dummy["positive_predictive_value"],
        0.5,
    )
    assert np.allclose(dummy["f1_score"], 2 / 3)
    assert np.allclose(
        dummy["balanced_accuracy"],
        0.5,
    )

    assert np.allclose(
        perfect["average_precision"],
        1.0,
    )
    assert np.allclose(perfect["roc_auc"], 1.0)
    assert np.allclose(perfect["sensitivity"], 1.0)
    assert np.allclose(perfect["specificity"], 1.0)


def test_repetition_summary_has_one_row_per_metric() -> None:
    """Summary rows must represent models and metrics."""
    repeat_metrics = compute_repeat_metrics(make_prediction_table())
    summary = summarize_repetitions(repeat_metrics)

    assert len(summary) == 2 * len(PERFORMANCE_METRICS)

    perfect_ap = summary[
        (summary["model"] == "perfect") & (summary["metric"] == "average_precision")
    ].iloc[0]

    assert perfect_ap["repeat_mean"] == pytest.approx(1.0)
    assert perfect_ap["repeat_standard_deviation"] == pytest.approx(0.0)
    assert perfect_ap["repeat_minimum"] == pytest.approx(1.0)
    assert perfect_ap["repeat_maximum"] == pytest.approx(1.0)
    assert perfect_ap["repetitions"] == 2


def test_grouped_bootstrap_is_reproducible() -> None:
    """The same seed must reproduce every interval."""
    predictions = make_prediction_table()

    first = grouped_bootstrap_intervals(
        predictions,
        n_resamples=25,
        random_state=101,
    )
    second = grouped_bootstrap_intervals(
        predictions,
        n_resamples=25,
        random_state=101,
    )

    assert_frame_equal(first, second)


def test_perfect_model_has_exact_bootstrap_limits() -> None:
    """Perfect ranking and classification remain perfect."""
    intervals = grouped_bootstrap_intervals(
        make_prediction_table(),
        n_resamples=25,
        random_state=103,
    )

    perfect = intervals[intervals["model"] == "perfect"]
    exact_metrics = perfect[
        perfect["metric"].isin(
            [
                "average_precision",
                "roc_auc",
                "sensitivity",
                "specificity",
                "positive_predictive_value",
                "f1_score",
                "balanced_accuracy",
            ]
        )
    ]

    assert np.allclose(
        exact_metrics["bootstrap_lower"],
        1.0,
    )
    assert np.allclose(
        exact_metrics["bootstrap_upper"],
        1.0,
    )


def test_build_and_save_summary_results(
    tmp_path: Path,
) -> None:
    """Summary construction must produce two reusable tables."""
    results = build_summary_results(
        make_prediction_table(),
        n_resamples=20,
        random_state=107,
    )

    assert len(results.repeat_metrics) == 4
    assert len(results.performance_summary) == (2 * len(PERFORMANCE_METRICS))
    assert (results.performance_summary["bootstrap_resamples"] == 20).all()

    paths = save_summary_results(
        results,
        reports_directory=tmp_path,
    )

    assert set(paths) == {
        "repeat_metrics",
        "performance_summary",
    }

    for path in paths.values():
        assert path.exists()
        assert "Unnamed: 0" not in pd.read_csv(path).columns


def test_prediction_validation_rejects_invalid_tables() -> None:
    """Missing columns and duplicate keys must fail."""
    predictions = make_prediction_table()

    with pytest.raises(ValueError, match="missing columns"):
        validate_prediction_table(predictions.drop(columns="group"))

    duplicated = pd.concat(
        [predictions, predictions.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="must be unique"):
        validate_prediction_table(duplicated)
