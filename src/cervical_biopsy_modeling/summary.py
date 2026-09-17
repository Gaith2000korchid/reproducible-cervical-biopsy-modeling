"""Performance summaries and grouped uncertainty intervals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cervical_biopsy_modeling.metrics import (
    compute_binary_metrics,
)

DEFAULT_PREDICTIONS_PATH = Path("reports/outer_predictions.csv")
DEFAULT_REPORTS_DIRECTORY = Path("reports")
REPEAT_METRICS_FILENAME = "repeat_metrics.csv"
PERFORMANCE_SUMMARY_FILENAME = "performance_summary.csv"

DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_RANDOM_STATE = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95

PERFORMANCE_METRICS = (
    "average_precision",
    "roc_auc",
    "brier_score",
    "sensitivity",
    "specificity",
    "positive_predictive_value",
    "f1_score",
    "balanced_accuracy",
)

REQUIRED_PREDICTION_COLUMNS = {
    "repeat",
    "model",
    "row_position",
    "group",
    "observed_outcome",
    "predicted_probability",
    "threshold",
}


@dataclass(frozen=True)
class AlignedPredictions:
    """Prediction arrays aligned across models and repetitions."""

    models: tuple[str, ...]
    repeats: tuple[int, ...]
    outcome: np.ndarray
    groups: np.ndarray
    probabilities: dict[tuple[str, int], np.ndarray]
    threshold: float


@dataclass(frozen=True)
class SummaryResults:
    """Repeat-level estimates and their final summary."""

    repeat_metrics: pd.DataFrame
    performance_summary: pd.DataFrame


def validate_prediction_table(
    predictions: pd.DataFrame,
) -> None:
    """Validate columns and uniqueness of outer predictions."""
    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction table is missing columns: {sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError("Prediction table must not be empty.")

    duplicate_keys = predictions.duplicated(subset=["model", "repeat", "row_position"])
    if duplicate_keys.any():
        raise ValueError("Each model, repeat, and row position must be unique.")

    if predictions["predicted_probability"].isna().any():
        raise ValueError("Predicted probabilities must not be missing.")

    if not predictions["predicted_probability"].between(0.0, 1.0).all():
        raise ValueError("Predicted probabilities must lie between 0 and 1.")


def align_predictions(
    predictions: pd.DataFrame,
) -> AlignedPredictions:
    """Align participant rows across models and repetitions."""
    validate_prediction_table(predictions)

    models = tuple(sorted(predictions["model"].unique()))
    repeats = tuple(sorted(int(value) for value in predictions["repeat"].unique()))

    reference = (
        predictions[
            (predictions["model"] == models[0]) & (predictions["repeat"] == repeats[0])
        ]
        .sort_values("row_position")
        .reset_index(drop=True)
    )

    reference_positions = reference["row_position"].to_numpy()
    outcome = reference["observed_outcome"].to_numpy(dtype=int)
    groups = reference["group"].to_numpy()

    thresholds = predictions["threshold"].unique()
    if len(thresholds) != 1:
        raise ValueError("All predictions must use the same threshold.")
    threshold = float(thresholds[0])

    probabilities: dict[
        tuple[str, int],
        np.ndarray,
    ] = {}

    for model in models:
        for repeat in repeats:
            current = (
                predictions[
                    (predictions["model"] == model) & (predictions["repeat"] == repeat)
                ]
                .sort_values("row_position")
                .reset_index(drop=True)
            )

            if not np.array_equal(
                current["row_position"].to_numpy(),
                reference_positions,
            ):
                raise ValueError(
                    "Models and repetitions must contain the same row positions."
                )

            if not np.array_equal(
                current["observed_outcome"].to_numpy(dtype=int),
                outcome,
            ):
                raise ValueError(
                    "Observed outcomes must agree across models and repetitions."
                )

            if not np.array_equal(
                current["group"].to_numpy(),
                groups,
            ):
                raise ValueError("Groups must agree across models and repetitions.")

            probabilities[(model, repeat)] = current["predicted_probability"].to_numpy(
                dtype=float
            )

    return AlignedPredictions(
        models=models,
        repeats=repeats,
        outcome=outcome,
        groups=groups,
        probabilities=probabilities,
        threshold=threshold,
    )


def compute_repeat_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Pool all out-of-fold predictions within each repeat."""
    aligned = align_predictions(predictions)
    records = []

    for model in aligned.models:
        for repeat in aligned.repeats:
            metrics = compute_binary_metrics(
                aligned.outcome,
                aligned.probabilities[(model, repeat)],
                threshold=aligned.threshold,
            )

            records.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "observations": len(aligned.outcome),
                    "positives": int(aligned.outcome.sum()),
                    **metrics.to_dict(),
                }
            )

    return pd.DataFrame.from_records(records)


def summarize_repetitions(
    repeat_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize performance variation across repetitions."""
    records = []

    for model in sorted(repeat_metrics["model"].unique()):
        model_rows = repeat_metrics[repeat_metrics["model"] == model]

        for metric in PERFORMANCE_METRICS:
            values = model_rows[metric].to_numpy(dtype=float)

            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "repeat_mean": float(values.mean()),
                    "repeat_standard_deviation": float(values.std(ddof=1)),
                    "repeat_minimum": float(values.min()),
                    "repeat_maximum": float(values.max()),
                    "repetitions": len(values),
                }
            )

    return pd.DataFrame.from_records(records)


def grouped_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Estimate paired intervals by resampling predictor groups."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least one.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one.")

    aligned = align_predictions(predictions)
    unique_groups = pd.unique(aligned.groups)
    group_positions = [
        np.flatnonzero(aligned.groups == group) for group in unique_groups
    ]

    distributions = {
        (model, metric): np.empty(
            n_resamples,
            dtype=float,
        )
        for model in aligned.models
        for metric in PERFORMANCE_METRICS
    }

    generator = np.random.default_rng(random_state)

    for bootstrap_index in range(n_resamples):
        for _ in range(100):
            sampled_group_indices = generator.integers(
                0,
                len(group_positions),
                size=len(group_positions),
            )
            sampled_positions = np.concatenate(
                [group_positions[index] for index in sampled_group_indices]
            )

            sampled_outcome = aligned.outcome[sampled_positions]
            if np.unique(sampled_outcome).size == 2:
                break
        else:
            raise RuntimeError(
                "Unable to draw a bootstrap sample containing both outcome classes."
            )

        for model in aligned.models:
            repeat_values = {metric: [] for metric in PERFORMANCE_METRICS}

            for repeat in aligned.repeats:
                metrics = compute_binary_metrics(
                    sampled_outcome,
                    aligned.probabilities[(model, repeat)][sampled_positions],
                    threshold=aligned.threshold,
                ).to_dict()

                for metric in PERFORMANCE_METRICS:
                    repeat_values[metric].append(float(metrics[metric]))

            for metric in PERFORMANCE_METRICS:
                distributions[(model, metric)][bootstrap_index] = float(
                    np.mean(repeat_values[metric])
                )

    alpha = (1.0 - confidence_level) / 2.0
    records = []

    for model in aligned.models:
        for metric in PERFORMANCE_METRICS:
            values = distributions[(model, metric)]

            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_lower": float(np.quantile(values, alpha)),
                    "bootstrap_upper": float(
                        np.quantile(
                            values,
                            1.0 - alpha,
                        )
                    ),
                    "bootstrap_resamples": n_resamples,
                    "bootstrap_random_state": random_state,
                    "confidence_level": confidence_level,
                    "resampling_unit": ("exact_predictor_group"),
                }
            )

    return pd.DataFrame.from_records(records)


def build_summary_results(
    predictions: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> SummaryResults:
    """Build repeat estimates and grouped intervals."""
    repeat_metrics = compute_repeat_metrics(predictions)
    repetition_summary = summarize_repetitions(repeat_metrics)
    intervals = grouped_bootstrap_intervals(
        predictions,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )

    performance_summary = repetition_summary.merge(
        intervals,
        on=["model", "metric"],
        how="inner",
        validate="one_to_one",
    )

    return SummaryResults(
        repeat_metrics=repeat_metrics,
        performance_summary=performance_summary,
    )


def save_summary_results(
    results: SummaryResults,
    *,
    reports_directory: Path = DEFAULT_REPORTS_DIRECTORY,
) -> dict[str, Path]:
    """Save repeat-level and summarized performance tables."""
    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "repeat_metrics": (reports_directory / REPEAT_METRICS_FILENAME),
        "performance_summary": (reports_directory / PERFORMANCE_SUMMARY_FILENAME),
    }

    results.repeat_metrics.to_csv(
        paths["repeat_metrics"],
        index=False,
    )
    results.performance_summary.to_csv(
        paths["performance_summary"],
        index=False,
    )

    return paths


def main() -> None:
    """Generate prespecified performance summaries."""
    predictions = pd.read_csv(DEFAULT_PREDICTIONS_PATH)

    results = build_summary_results(predictions)
    paths = save_summary_results(results)

    print(
        "Repeat-level metric rows:",
        len(results.repeat_metrics),
    )
    print(
        "Performance summary rows:",
        len(results.performance_summary),
    )

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
