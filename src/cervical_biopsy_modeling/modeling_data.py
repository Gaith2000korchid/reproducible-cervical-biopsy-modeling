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


def _build_modeling_dataset(
    data: pd.DataFrame,
    predictor_names: list[str],
) -> ModelingDataset:
    """Build modeling inputs from an explicitly selected predictor set."""

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

    # Groups deliberately remain based on the primary pre-examination
    # predictors. This preserves duplicate handling and permits paired
    # comparisons between the primary and diagnostic-inclusive analyses.
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


def build_modeling_dataset(
    data: pd.DataFrame,
) -> ModelingDataset:
    """Build the prespecified leakage-aware modeling inputs."""

    predictor_names = primary_predictor_columns(data)

    return _build_modeling_dataset(
        data,
        predictor_names,
    )


def build_diagnostic_inclusive_modeling_dataset(
    data: pd.DataFrame,
) -> ModelingDataset:
    """Build inputs for the explicitly labelled leakage audit.

    All seven current-examination or existing-diagnosis variables are
    intentionally retained. These inputs must not be interpreted as
    valid pre-screening predictors.
    """

    # Validate that the outcome and all prespecified leakage columns exist.
    primary_predictor_columns(data)

    predictor_names = [column for column in data.columns if column != OUTCOME_COLUMN]

    return _build_modeling_dataset(
        data,
        predictor_names,
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
