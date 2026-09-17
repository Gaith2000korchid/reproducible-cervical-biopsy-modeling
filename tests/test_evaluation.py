"""Tests for nested outer-fold evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cervical_biopsy_modeling.evaluation import (
    EvaluationResults,
    evaluate_outer_splits,
    save_evaluation_results,
)
from cervical_biopsy_modeling.modeling_data import (
    ModelingDataset,
)
from cervical_biopsy_modeling.models import (
    build_dummy_specification,
)
from cervical_biopsy_modeling.validation import (
    build_repeated_group_splits,
)


def make_grouped_modeling_dataset() -> ModelingDataset:
    """Create 12 homogeneous groups with repeated rows."""
    group_values = np.repeat(np.arange(12), 2)
    group_outcomes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    outcome_values = np.repeat(group_outcomes, 2)

    predictors = pd.DataFrame(
        {
            "continuous": np.linspace(0.0, 10.0, 24),
            "binary": np.tile([0, 1, 0, 1], 6),
            "partly_missing": [
                np.nan if index % 5 == 0 else float(index) for index in range(24)
            ],
            "constant": [1] * 24,
        }
    )
    outcome = pd.Series(
        outcome_values,
        name="Biopsy",
        dtype=int,
    )
    groups = pd.Series(
        group_values,
        name="predictor_group",
        dtype=int,
    )

    return ModelingDataset(
        predictors=predictors,
        outcome=outcome,
        groups=groups,
    )


@pytest.fixture(scope="module")
def dummy_evaluation() -> EvaluationResults:
    """Run a compact nested evaluation once for this module."""
    modeling = make_grouped_modeling_dataset()
    splits = build_repeated_group_splits(
        modeling.predictors,
        modeling.outcome,
        modeling.groups,
        n_splits=4,
        n_repeats=1,
        random_state=19,
    )
    specification = build_dummy_specification()

    return evaluate_outer_splits(
        modeling,
        splits,
        {specification.name: specification},
        inner_n_splits=3,
        threshold=0.5,
        n_jobs=1,
    )


def test_evaluation_builds_expected_table_sizes(
    dummy_evaluation: EvaluationResults,
) -> None:
    """One model over four folds must yield stable row counts."""
    assert len(dummy_evaluation.fold_metrics) == 4
    assert len(dummy_evaluation.predictions) == 24
    assert len(dummy_evaluation.selected_parameters) == 4
    assert len(dummy_evaluation.inner_candidates) == 4

    assert set(dummy_evaluation.fold_metrics["model"]) == {"dummy_prior"}
    assert set(dummy_evaluation.predictions["model"]) == {"dummy_prior"}


def test_dummy_outer_metrics_match_prevalence(
    dummy_evaluation: EvaluationResults,
) -> None:
    """Constant probabilities must reproduce baseline metrics."""
    metrics = dummy_evaluation.fold_metrics

    assert np.allclose(
        metrics["average_precision"],
        1 / 3,
    )
    assert np.allclose(metrics["roc_auc"], 0.5)
    assert np.allclose(metrics["brier_score"], 2 / 9)
    assert np.allclose(metrics["sensitivity"], 0.0)
    assert np.allclose(metrics["specificity"], 1.0)
    assert np.allclose(
        metrics["balanced_accuracy"],
        0.5,
    )
    assert (metrics["true_negatives"] == 4).all()
    assert (metrics["false_positives"] == 0).all()
    assert (metrics["false_negatives"] == 2).all()
    assert (metrics["true_positives"] == 0).all()


def test_predictions_cover_each_row_once(
    dummy_evaluation: EvaluationResults,
) -> None:
    """One repetition must predict every observation once."""
    predictions = dummy_evaluation.predictions

    counts = predictions.groupby("row_position").size()

    assert counts.index.tolist() == list(range(24))
    assert (counts == 1).all()
    assert np.allclose(
        predictions["predicted_probability"],
        1 / 3,
    )
    assert (predictions["predicted_class"] == 0).all()

    ordered = predictions.sort_values("row_position")
    expected_outcome = make_grouped_modeling_dataset().outcome.to_numpy()
    assert np.array_equal(
        ordered["observed_outcome"].to_numpy(),
        expected_outcome,
    )


def test_selected_parameters_record_fitted_pipeline(
    dummy_evaluation: EvaluationResults,
) -> None:
    """Selection records must describe the fitted estimator."""
    selected = dummy_evaluation.selected_parameters

    assert (selected["best_parameters"] == "{}").all()
    assert (selected["inner_candidate_count"] == 1).all()
    assert (selected["processed_predictor_count"] > 0).all()
    assert selected["fitted_scale_pos_weight"].isna().all()


def test_save_evaluation_results_writes_all_tables(
    dummy_evaluation: EvaluationResults,
    tmp_path: Path,
) -> None:
    """Every result table must be saved without its index."""
    paths = save_evaluation_results(
        dummy_evaluation,
        reports_directory=tmp_path,
    )

    assert set(paths) == {
        "fold_metrics",
        "predictions",
        "selected_parameters",
        "inner_candidates",
    }

    expected_lengths = {
        "fold_metrics": 4,
        "predictions": 24,
        "selected_parameters": 4,
        "inner_candidates": 4,
    }

    for name, path in paths.items():
        assert path.exists()
        table = pd.read_csv(path)
        assert len(table) == expected_lengths[name]
        assert "Unnamed: 0" not in table.columns


def test_evaluation_requires_splits_and_models() -> None:
    """Empty evaluation inputs must fail explicitly."""
    modeling = make_grouped_modeling_dataset()
    specification = build_dummy_specification()
    splits = build_repeated_group_splits(
        modeling.predictors,
        modeling.outcome,
        modeling.groups,
        n_splits=4,
        n_repeats=1,
        random_state=23,
    )

    with pytest.raises(ValueError, match="outer split"):
        evaluate_outer_splits(
            modeling,
            [],
            {specification.name: specification},
        )

    with pytest.raises(ValueError, match="model specification"):
        evaluate_outer_splits(
            modeling,
            splits,
            {},
        )
