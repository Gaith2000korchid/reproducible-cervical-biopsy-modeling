"""Tests for group-aware inner model tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cervical_biopsy_modeling.models import (
    LOGISTIC_C_VALUES,
    build_dummy_specification,
    build_logistic_specification,
)
from cervical_biopsy_modeling.tuning import (
    build_inner_splits,
    tune_model,
)


def make_grouped_classification_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create balanced groups with a minority positive outcome."""
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

    return predictors, outcome, groups


def test_inner_splits_are_group_disjoint_and_exhaustive() -> None:
    """Every row must be tested once without group overlap."""
    predictors, outcome, groups = make_grouped_classification_data()

    splits = build_inner_splits(
        predictors,
        outcome,
        groups,
        n_splits=4,
        random_state=17,
    )

    assert len(splits) == 4

    observed_test_indices = []

    for train_indices, test_indices in splits:
        train_groups = set(groups.iloc[train_indices])
        test_groups = set(groups.iloc[test_indices])

        assert train_groups.isdisjoint(test_groups)
        assert outcome.iloc[test_indices].sum() == 2
        observed_test_indices.extend(test_indices.tolist())

    assert sorted(observed_test_indices) == list(range(len(predictors)))


def test_inner_splits_are_reproducible() -> None:
    """The same seed must reproduce the same folds."""
    predictors, outcome, groups = make_grouped_classification_data()

    first = build_inner_splits(
        predictors,
        outcome,
        groups,
        random_state=29,
    )
    second = build_inner_splits(
        predictors,
        outcome,
        groups,
        random_state=29,
    )

    for first_split, second_split in zip(
        first,
        second,
        strict=True,
    ):
        assert np.array_equal(
            first_split[0],
            second_split[0],
        )
        assert np.array_equal(
            first_split[1],
            second_split[1],
        )


def test_dummy_tuning_returns_fitted_reference() -> None:
    """The dummy model must complete inner validation."""
    predictors, outcome, groups = make_grouped_classification_data()

    result = tune_model(
        build_dummy_specification(),
        predictors,
        outcome,
        groups,
        random_state=31,
    )

    probabilities = result.best_estimator.predict_proba(predictors)[:, 1]

    assert result.model_name == "dummy_prior"
    assert result.best_parameters == {}
    assert result.best_inner_average_precision == pytest.approx(outcome.mean())
    assert len(result.candidate_results) == 1
    assert np.allclose(probabilities, outcome.mean())


def test_logistic_tuning_evaluates_all_c_values() -> None:
    """Every prespecified regularization value must be evaluated."""
    predictors, outcome, groups = make_grouped_classification_data()

    result = tune_model(
        build_logistic_specification(random_state=37),
        predictors,
        outcome,
        groups,
        random_state=37,
    )

    probabilities = result.best_estimator.predict_proba(predictors)[:, 1]

    assert result.model_name == "logistic_regression"
    assert result.best_parameters["classifier__C"] in LOGISTIC_C_VALUES
    assert len(result.candidate_results) == len(LOGISTIC_C_VALUES)
    assert result.candidate_results["candidate"].tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert result.candidate_results["rank_inner_average_precision"].min() == 1
    assert probabilities.shape == (len(predictors),)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_candidate_table_has_stable_schema() -> None:
    """The candidate summary must be ready for concatenation."""
    predictors, outcome, groups = make_grouped_classification_data()

    result = tune_model(
        build_dummy_specification(),
        predictors,
        outcome,
        groups,
        random_state=41,
    )

    assert result.candidate_results.columns.tolist() == [
        "model",
        "candidate",
        "parameters",
        "mean_inner_average_precision",
        "std_inner_average_precision",
        "rank_inner_average_precision",
    ]
