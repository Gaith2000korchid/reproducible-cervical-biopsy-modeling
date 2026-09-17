"""Build leakage-aware modeling inputs from validated raw data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cervical_biopsy_modeling.audit import (
    OUTCOME_COLUMN,
    make_predictor_groups,
    primary_predictor_columns,
)
from cervical_biopsy_modeling.data import (
    DatasetValidationError,
)


@dataclass(frozen=True)
class ModelingDataset:
    """Predictors, outcome, and groups used for modeling."""

    predictors: pd.DataFrame
    outcome: pd.Series
    groups: pd.Series


def build_modeling_dataset(
    data: pd.DataFrame,
) -> ModelingDataset:
    """Build inputs without fitting preprocessing operations."""
    predictor_names = primary_predictor_columns(data)
    predictors = data.loc[:, predictor_names].copy()
    outcome = data.loc[:, OUTCOME_COLUMN].copy()

    if outcome.isna().any():
        raise DatasetValidationError("The modeling outcome contains missing values.")

    observed_values = set(outcome.unique().tolist())

    if observed_values != {0.0, 1.0}:
        raise DatasetValidationError(
            "The modeling outcome must contain exactly the values 0 and 1."
        )

    outcome = outcome.astype("int8").rename(OUTCOME_COLUMN)
    groups = make_predictor_groups(data)

    if not predictors.index.equals(outcome.index):
        raise DatasetValidationError("Predictors and outcome indices do not match.")

    if not predictors.index.equals(groups.index):
        raise DatasetValidationError("Predictors and group indices do not match.")

    return ModelingDataset(
        predictors=predictors,
        outcome=outcome,
        groups=groups,
    )


def retain_first_observation_per_group(
    modeling: ModelingDataset,
) -> ModelingDataset:
    """Retain the first row of each exact-predictor group.

    Selection follows the existing row order and does not inspect
    the outcome. Indices are reset for subsequent positional
    cross-validation.
    """
    retained = ~modeling.groups.duplicated(keep="first")

    predictors = modeling.predictors.loc[retained].reset_index(drop=True)
    outcome = modeling.outcome.loc[retained].reset_index(drop=True)
    groups = modeling.groups.loc[retained].reset_index(drop=True)

    if not groups.is_unique:
        raise DatasetValidationError("Deduplicated modeling groups must be unique.")

    return ModelingDataset(
        predictors=predictors,
        outcome=outcome,
        groups=groups,
    )
