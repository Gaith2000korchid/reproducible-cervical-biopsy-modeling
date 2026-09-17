"""Group-aware inner validation for model tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from cervical_biopsy_modeling.models import ModelSpecification
from cervical_biopsy_modeling.validation import (
    build_repeated_group_splits,
)

DEFAULT_INNER_SPLITS = 4
PRIMARY_SCORING = "average_precision"


@dataclass(frozen=True)
class TuningResult:
    """Result of group-aware inner model selection."""

    model_name: str
    best_estimator: Pipeline
    best_parameters: dict[str, Any]
    best_inner_average_precision: float
    candidate_results: pd.DataFrame


def build_inner_splits(
    predictors: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = DEFAULT_INNER_SPLITS,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build one repetition of stratified group-aware inner folds."""
    grouped_splits = build_repeated_group_splits(
        predictors,
        outcome,
        groups,
        n_splits=n_splits,
        n_repeats=1,
        random_state=random_state,
    )

    return [(split.train_indices, split.test_indices) for split in grouped_splits]


def summarize_search(
    search: GridSearchCV,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Create a compact table of inner-validation candidates."""
    records = []

    for candidate_index, parameters in enumerate(search.cv_results_["params"]):
        records.append(
            {
                "model": model_name,
                "candidate": candidate_index + 1,
                "parameters": json.dumps(
                    parameters,
                    sort_keys=True,
                ),
                "mean_inner_average_precision": float(
                    search.cv_results_["mean_test_score"][candidate_index]
                ),
                "std_inner_average_precision": float(
                    search.cv_results_["std_test_score"][candidate_index]
                ),
                "rank_inner_average_precision": int(
                    search.cv_results_["rank_test_score"][candidate_index]
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def tune_model(
    specification: ModelSpecification,
    predictors: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = DEFAULT_INNER_SPLITS,
    random_state: int = 42,
    n_jobs: int = 1,
) -> TuningResult:
    """Select hyperparameters using group-aware inner validation."""
    inner_splits = build_inner_splits(
        predictors,
        outcome,
        groups,
        n_splits=n_splits,
        random_state=random_state,
    )

    search = GridSearchCV(
        estimator=specification.estimator,
        param_grid=specification.parameter_grid,
        scoring=PRIMARY_SCORING,
        cv=inner_splits,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(predictors, outcome)

    return TuningResult(
        model_name=specification.name,
        best_estimator=search.best_estimator_,
        best_parameters=dict(search.best_params_),
        best_inner_average_precision=float(search.best_score_),
        candidate_results=summarize_search(
            search,
            model_name=specification.name,
        ),
    )
