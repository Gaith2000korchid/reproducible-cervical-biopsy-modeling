"""Tests for leakage-aware modeling inputs."""

from __future__ import annotations

import pandas as pd
import pytest

from cervical_biopsy_modeling.audit import LEAKAGE_COLUMNS
from cervical_biopsy_modeling.data import DatasetValidationError
from cervical_biopsy_modeling.modeling_data import build_modeling_dataset


def make_modeling_dataframe() -> pd.DataFrame:
    """Create a small dataset with one duplicated predictor profile."""
    return pd.DataFrame(
        {
            "Age": [20, 20, 30],
            "First sexual intercourse": [18, 18, 19],
            "risk_feature": [2.0, 2.0, 1.0],
            "nullable_feature": [float("nan"), float("nan"), 0.0],
            "Hinselmann": [0, 1, 0],
            "Schiller": [0, 1, 0],
            "Citology": [0, 1, 0],
            "Dx:Cancer": [0, 0, 0],
            "Dx:CIN": [0, 0, 0],
            "Dx:HPV": [0, 0, 0],
            "Dx": [0, 0, 0],
            "Biopsy": [0, 1, 0],
        }
    )


def test_build_modeling_dataset_excludes_leakage_columns() -> None:
    modeling_data = build_modeling_dataset(make_modeling_dataframe())

    assert modeling_data.predictors.columns.tolist() == [
        "Age",
        "First sexual intercourse",
        "risk_feature",
        "nullable_feature",
    ]
    assert set(LEAKAGE_COLUMNS).isdisjoint(modeling_data.predictors.columns)
    assert modeling_data.outcome.tolist() == [0, 1, 0]
    assert str(modeling_data.outcome.dtype) == "int8"


def test_build_modeling_dataset_preserves_duplicate_groups() -> None:
    modeling_data = build_modeling_dataset(make_modeling_dataframe())

    assert modeling_data.groups.iloc[0] == modeling_data.groups.iloc[1]
    assert modeling_data.groups.iloc[0] != modeling_data.groups.iloc[2]
    assert modeling_data.groups.nunique() == 2


def test_build_modeling_dataset_rejects_missing_outcome() -> None:
    data = make_modeling_dataframe()
    data.loc[0, "Biopsy"] = float("nan")

    with pytest.raises(DatasetValidationError, match="missing"):
        build_modeling_dataset(data)


def test_build_modeling_dataset_rejects_nonbinary_outcome() -> None:
    data = make_modeling_dataframe()
    data.loc[0, "Biopsy"] = 2

    with pytest.raises(DatasetValidationError, match="exactly"):
        build_modeling_dataset(data)
