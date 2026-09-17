"""Nested outer-fold evaluation and reproducible result tables."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cervical_biopsy_modeling.audit import (
    load_validated_dataset,
)
from cervical_biopsy_modeling.metrics import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    compute_binary_metrics,
)
from cervical_biopsy_modeling.modeling_data import (
    ModelingDataset,
    build_modeling_dataset,
)
from cervical_biopsy_modeling.models import (
    ModelSpecification,
    build_model_specifications,
)
from cervical_biopsy_modeling.tuning import tune_model
from cervical_biopsy_modeling.validation import (
    GroupedSplit,
    build_repeated_group_splits,
)

DEFAULT_REPORTS_DIRECTORY = Path("reports")
FOLD_METRICS_FILENAME = "outer_fold_metrics.csv"
PREDICTIONS_FILENAME = "outer_predictions.csv"
SELECTED_PARAMETERS_FILENAME = "selected_hyperparameters.csv"
INNER_CANDIDATES_FILENAME = "inner_cv_candidates.csv"


@dataclass(frozen=True)
class EvaluationResults:
    """All tabular outputs from nested outer evaluation."""

    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_parameters: pd.DataFrame
    inner_candidates: pd.DataFrame


def _processed_feature_count(estimator: Any) -> int:
    """Count features produced by the fitted preprocessor."""
    preprocessing = estimator.named_steps["preprocessing"]
    feature_names = preprocessing.get_feature_names_out()
    return int(len(feature_names))


def _fitted_scale_pos_weight(estimator: Any) -> float | None:
    """Return the fitted XGBoost class weight when available."""
    classifier = estimator.named_steps["classifier"]
    value = getattr(
        classifier,
        "scale_pos_weight_",
        None,
    )

    if value is None:
        return None

    return float(value)


def evaluate_outer_splits(
    modeling: ModelingDataset,
    outer_splits: Sequence[GroupedSplit],
    specifications: Mapping[str, ModelSpecification],
    *,
    inner_n_splits: int = 4,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    n_jobs: int = 1,
) -> EvaluationResults:
    """Tune and evaluate every model on every outer fold."""
    if not outer_splits:
        raise ValueError("At least one outer split is required.")

    if not specifications:
        raise ValueError("At least one model specification is required.")

    metric_records: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    selected_parameter_records: list[dict[str, Any]] = []
    inner_candidate_tables: list[pd.DataFrame] = []

    for split in outer_splits:
        train_positions = np.asarray(split.train_indices)
        test_positions = np.asarray(split.test_indices)

        X_train = modeling.predictors.iloc[train_positions]
        X_test = modeling.predictors.iloc[test_positions]
        y_train = modeling.outcome.iloc[train_positions]
        y_test = modeling.outcome.iloc[test_positions]
        groups_train = modeling.groups.iloc[train_positions]
        groups_test = modeling.groups.iloc[test_positions]

        for specification in specifications.values():
            tuning = tune_model(
                specification,
                X_train,
                y_train,
                groups_train,
                n_splits=inner_n_splits,
                random_state=split.random_state,
                n_jobs=n_jobs,
            )

            probabilities = tuning.best_estimator.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            metrics = compute_binary_metrics(
                y_test,
                probabilities,
                threshold=threshold,
            )

            metric_records.append(
                {
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "random_state": split.random_state,
                    "model": specification.name,
                    "train_size": len(train_positions),
                    "test_size": len(test_positions),
                    "train_positives": int(y_train.sum()),
                    "test_positives": int(y_test.sum()),
                    **metrics.to_dict(),
                }
            )

            prediction_tables.append(
                pd.DataFrame(
                    {
                        "repeat": split.repeat,
                        "fold": split.fold,
                        "random_state": split.random_state,
                        "model": specification.name,
                        "row_position": test_positions,
                        "group": groups_test.to_numpy(),
                        "observed_outcome": (y_test.to_numpy(dtype=int)),
                        "predicted_probability": probabilities,
                        "predicted_class": predictions,
                        "threshold": float(threshold),
                    }
                )
            )

            selected_parameter_records.append(
                {
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "random_state": split.random_state,
                    "model": specification.name,
                    "best_parameters": json.dumps(
                        tuning.best_parameters,
                        sort_keys=True,
                    ),
                    "best_inner_average_precision": (
                        tuning.best_inner_average_precision
                    ),
                    "inner_candidate_count": len(tuning.candidate_results),
                    "processed_predictor_count": (
                        _processed_feature_count(tuning.best_estimator)
                    ),
                    "fitted_scale_pos_weight": (
                        _fitted_scale_pos_weight(tuning.best_estimator)
                    ),
                }
            )

            candidate_table = tuning.candidate_results.copy()
            candidate_table.insert(
                0,
                "random_state",
                split.random_state,
            )
            candidate_table.insert(0, "fold", split.fold)
            candidate_table.insert(
                0,
                "repeat",
                split.repeat,
            )
            inner_candidate_tables.append(candidate_table)

    return EvaluationResults(
        fold_metrics=pd.DataFrame.from_records(metric_records),
        predictions=pd.concat(
            prediction_tables,
            ignore_index=True,
        ),
        selected_parameters=pd.DataFrame.from_records(selected_parameter_records),
        inner_candidates=pd.concat(
            inner_candidate_tables,
            ignore_index=True,
        ),
    )


def save_evaluation_results(
    results: EvaluationResults,
    *,
    reports_directory: Path = DEFAULT_REPORTS_DIRECTORY,
) -> dict[str, Path]:
    """Write all nested-evaluation tables to CSV."""
    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "fold_metrics": (reports_directory / FOLD_METRICS_FILENAME),
        "predictions": (reports_directory / PREDICTIONS_FILENAME),
        "selected_parameters": (reports_directory / SELECTED_PARAMETERS_FILENAME),
        "inner_candidates": (reports_directory / INNER_CANDIDATES_FILENAME),
    }

    results.fold_metrics.to_csv(
        paths["fold_metrics"],
        index=False,
    )
    results.predictions.to_csv(
        paths["predictions"],
        index=False,
    )
    results.selected_parameters.to_csv(
        paths["selected_parameters"],
        index=False,
    )
    results.inner_candidates.to_csv(
        paths["inner_candidates"],
        index=False,
    )

    return paths


def main() -> None:
    """Run the complete prespecified nested evaluation."""
    data = load_validated_dataset()
    modeling = build_modeling_dataset(data)

    outer_splits = build_repeated_group_splits(
        modeling.predictors,
        modeling.outcome,
        modeling.groups,
    )
    specifications = build_model_specifications()

    results = evaluate_outer_splits(
        modeling,
        outer_splits,
        specifications,
        inner_n_splits=4,
        threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        n_jobs=1,
    )
    paths = save_evaluation_results(results)

    print(f"Evaluated outer folds: {len(outer_splits)}")
    print(f"Evaluated models: {len(specifications)}")
    print(
        "Fold metric rows:",
        len(results.fold_metrics),
    )
    print(
        "Prediction rows:",
        len(results.predictions),
    )
    print(
        "Selected-parameter rows:",
        len(results.selected_parameters),
    )
    print(
        "Inner-candidate rows:",
        len(results.inner_candidates),
    )

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
