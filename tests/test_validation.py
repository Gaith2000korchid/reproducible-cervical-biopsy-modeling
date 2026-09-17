"""Tests for repeated stratified group-aware validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cervical_biopsy_modeling.validation import (
    build_repeated_group_splits,
    summarize_splits,
)


def make_validation_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create 20 duplicated groups balanced across two outcome classes."""
    groups = pd.Series(
        np.repeat(np.arange(20), 2),
        name="predictor_group",
    )
    outcome = pd.Series(
        np.repeat(np.tile([0, 1], 10), 2),
        name="Biopsy",
        dtype="int8",
    )
    predictors = pd.DataFrame(
        {
            "feature_1": np.arange(40),
            "feature_2": np.repeat(np.arange(20), 2),
        }
    )
    return predictors, outcome, groups


def test_repeated_splits_have_expected_count_and_isolated_groups() -> None:
    predictors, outcome, groups = make_validation_data()

    splits = build_repeated_group_splits(
        predictors,
        outcome,
        groups,
        n_splits=5,
        n_repeats=3,
        random_state=42,
    )

    assert len(splits) == 15

    for split in splits:
        train_groups = set(groups.iloc[split.train_indices])
        test_groups = set(groups.iloc[split.test_indices])

        assert train_groups.isdisjoint(test_groups)
        assert outcome.iloc[split.train_indices].nunique() == 2
        assert outcome.iloc[split.test_indices].nunique() == 2


def test_every_observation_is_tested_once_per_repeat() -> None:
    predictors, outcome, groups = make_validation_data()

    splits = build_repeated_group_splits(
        predictors,
        outcome,
        groups,
        n_splits=5,
        n_repeats=2,
        random_state=42,
    )

    for repeat in (1, 2):
        repeat_indices = np.concatenate(
            [split.test_indices for split in splits if split.repeat == repeat]
        )
        assert np.array_equal(
            np.sort(repeat_indices),
            np.arange(len(predictors)),
        )


def test_repeated_splits_are_reproducible() -> None:
    predictors, outcome, groups = make_validation_data()

    first = build_repeated_group_splits(
        predictors,
        outcome,
        groups,
        n_splits=5,
        n_repeats=2,
        random_state=42,
    )
    second = build_repeated_group_splits(
        predictors,
        outcome,
        groups,
        n_splits=5,
        n_repeats=2,
        random_state=42,
    )

    for first_split, second_split in zip(first, second, strict=True):
        assert first_split.repeat == second_split.repeat
        assert first_split.fold == second_split.fold
        assert first_split.random_state == second_split.random_state
        assert np.array_equal(
            first_split.train_indices,
            second_split.train_indices,
        )
        assert np.array_equal(
            first_split.test_indices,
            second_split.test_indices,
        )


def test_split_summary_contains_one_row_per_fold() -> None:
    predictors, outcome, groups = make_validation_data()
    splits = build_repeated_group_splits(
        predictors,
        outcome,
        groups,
        n_splits=5,
        n_repeats=2,
        random_state=42,
    )

    summary = summarize_splits(splits, outcome, groups)

    assert summary.shape[0] == 10
    assert summary["repeat"].nunique() == 2
    assert summary.groupby("repeat")["fold"].nunique().eq(5).all()
    assert summary["test_positives"].gt(0).all()
    assert summary["test_negatives"].gt(0).all()


def test_repeated_splits_reject_misaligned_indices() -> None:
    predictors, outcome, groups = make_validation_data()
    outcome.index = outcome.index + 1

    with pytest.raises(ValueError, match="indices do not match"):
        build_repeated_group_splits(
            predictors,
            outcome,
            groups,
        )
