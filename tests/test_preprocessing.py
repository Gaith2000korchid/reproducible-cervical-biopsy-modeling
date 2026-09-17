"""Tests for leakage-safe preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cervical_biopsy_modeling.preprocessing import (
    MissingnessFilter,
    build_preprocessing_pipeline,
)


def test_missingness_filter_keeps_exact_threshold() -> None:
    """A feature with exactly 80% missingness must be retained."""
    data = pd.DataFrame(
        {
            "complete": [1, 2, 3, 4, 5],
            "at_threshold": [1, np.nan, np.nan, np.nan, np.nan],
            "above_threshold": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    transformer = MissingnessFilter(max_missing_fraction=0.80)
    transformed = transformer.fit_transform(data)

    assert transformed.columns.tolist() == [
        "complete",
        "at_threshold",
    ]


def test_missingness_filter_uses_training_missingness() -> None:
    """Test-fold missingness must not change training-fold selection."""
    training_data = pd.DataFrame(
        {
            "retained": [1, 2, 3, 4, 5],
            "removed": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    test_data = pd.DataFrame(
        {
            "retained": [np.nan, np.nan],
            "removed": [10, 20],
        }
    )

    transformer = MissingnessFilter(max_missing_fraction=0.80)
    transformer.fit(training_data)
    transformed = transformer.transform(test_data)

    assert transformed.columns.tolist() == ["retained"]
    assert transformed.index.equals(test_data.index)


def test_missingness_filter_rejects_reordered_columns() -> None:
    """Transform columns must match the fitted names and order."""
    training_data = pd.DataFrame(
        {
            "first": [1, 2, 3],
            "second": [4, 5, 6],
        }
    )

    transformer = MissingnessFilter()
    transformer.fit(training_data)

    reordered_data = training_data.loc[:, ["second", "first"]]

    with pytest.raises(ValueError, match="columns and order"):
        transformer.transform(reordered_data)


def test_preprocessing_pipeline_imputes_and_filters() -> None:
    """The pipeline must impute values and remove unusable features."""
    data = pd.DataFrame(
        {
            "continuous": [1, 2, np.nan, 4, 5, 6],
            "constant": [7, 7, 7, 7, 7, 7],
            "partly_missing": [1, np.nan, 3, np.nan, 5, 6],
            "too_missing": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )

    pipeline = build_preprocessing_pipeline(scale=False)
    transformed = pipeline.fit_transform(data)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == data.shape[0]
    assert not transformed.isna().any().any()
    assert "constant" not in transformed.columns
    assert "too_missing" not in transformed.columns
    assert any(column.startswith("missingindicator_") for column in transformed.columns)


def test_scaled_pipeline_centers_retained_features() -> None:
    """Scaling must be applied after the other preprocessing steps."""
    data = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 4.0],
            "second": [4.0, 6.0, 8.0, 10.0],
        }
    )

    pipeline = build_preprocessing_pipeline(scale=True)
    transformed = pipeline.fit_transform(data)

    assert isinstance(transformed, pd.DataFrame)
    assert np.allclose(transformed.mean().to_numpy(), 0.0)
