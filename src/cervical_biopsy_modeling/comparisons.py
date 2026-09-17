"""Paired grouped-bootstrap comparisons between models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cervical_biopsy_modeling.metrics import (
    compute_binary_metrics,
)
from cervical_biopsy_modeling.summary import (
    DEFAULT_BOOTSTRAP_RANDOM_STATE,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_PREDICTIONS_PATH,
    PERFORMANCE_METRICS,
    align_predictions,
    compute_repeat_metrics,
)

DEFAULT_REPORTS_DIRECTORY = Path("reports")
COMPARISONS_FILENAME = "paired_model_comparisons.csv"

DEFAULT_MODEL_COMPARISONS = (
    ("logistic_regression", "dummy_prior"),
    ("xgboost", "dummy_prior"),
    ("logistic_regression", "xgboost"),
)


def oriented_difference(
    candidate_value: float,
    reference_value: float,
    *,
    metric: str,
) -> float:
    """Return a difference where positive means candidate is better."""
    if metric == "brier_score":
        return reference_value - candidate_value

    return candidate_value - reference_value


def _validate_comparisons(
    available_models: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
) -> None:
    """Require distinct comparison models present in predictions."""
    available = set(available_models)

    if not comparisons:
        raise ValueError("At least one model comparison is required.")

    for candidate, reference in comparisons:
        if candidate == reference:
            raise ValueError("Candidate and reference models must differ.")

        missing = {model for model in (candidate, reference) if model not in available}
        if missing:
            raise ValueError(
                f"Comparison contains unavailable models: {sorted(missing)}"
            )


def compute_repeat_differences(
    predictions: pd.DataFrame,
    *,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_MODEL_COMPARISONS,
) -> pd.DataFrame:
    """Compute paired differences within complete repetitions."""
    aligned = align_predictions(predictions)
    _validate_comparisons(
        aligned.models,
        comparisons,
    )

    repeat_metrics = compute_repeat_metrics(predictions)
    records = []

    for candidate, reference in comparisons:
        for metric in PERFORMANCE_METRICS:
            differences = []

            for repeat in aligned.repeats:
                candidate_value = float(
                    repeat_metrics.loc[
                        (repeat_metrics["model"] == candidate)
                        & (repeat_metrics["repeat"] == repeat),
                        metric,
                    ].iloc[0]
                )
                reference_value = float(
                    repeat_metrics.loc[
                        (repeat_metrics["model"] == reference)
                        & (repeat_metrics["repeat"] == repeat),
                        metric,
                    ].iloc[0]
                )

                differences.append(
                    oriented_difference(
                        candidate_value,
                        reference_value,
                        metric=metric,
                    )
                )

            difference_array = np.asarray(
                differences,
                dtype=float,
            )

            records.append(
                {
                    "candidate_model": candidate,
                    "reference_model": reference,
                    "metric": metric,
                    "repeat_mean_difference": float(difference_array.mean()),
                    "repeat_standard_deviation": float(difference_array.std(ddof=1)),
                    "repeat_minimum_difference": float(difference_array.min()),
                    "repeat_maximum_difference": float(difference_array.max()),
                    "repetitions": len(difference_array),
                    "difference_definition": (
                        "reference_minus_candidate"
                        if metric == "brier_score"
                        else "candidate_minus_reference"
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


def grouped_bootstrap_comparisons(
    predictions: pd.DataFrame,
    *,
    comparisons: Sequence[tuple[str, str]] = DEFAULT_MODEL_COMPARISONS,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_BOOTSTRAP_RANDOM_STATE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Estimate paired grouped-bootstrap difference intervals."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least one.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one.")

    aligned = align_predictions(predictions)
    _validate_comparisons(
        aligned.models,
        comparisons,
    )

    point_estimates = compute_repeat_differences(
        predictions,
        comparisons=comparisons,
    )

    unique_groups = pd.unique(aligned.groups)
    group_positions = [
        np.flatnonzero(aligned.groups == group) for group in unique_groups
    ]

    distributions = {
        (candidate, reference, metric): np.empty(
            n_resamples,
            dtype=float,
        )
        for candidate, reference in comparisons
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

        repeat_differences = {
            (candidate, reference, metric): []
            for candidate, reference in comparisons
            for metric in PERFORMANCE_METRICS
        }

        for repeat in aligned.repeats:
            metrics_by_model = {}

            for model in aligned.models:
                metrics_by_model[model] = compute_binary_metrics(
                    sampled_outcome,
                    aligned.probabilities[(model, repeat)][sampled_positions],
                    threshold=aligned.threshold,
                ).to_dict()

            for candidate, reference in comparisons:
                for metric in PERFORMANCE_METRICS:
                    repeat_differences[(candidate, reference, metric)].append(
                        oriented_difference(
                            float(metrics_by_model[candidate][metric]),
                            float(metrics_by_model[reference][metric]),
                            metric=metric,
                        )
                    )

        for key, values in repeat_differences.items():
            distributions[key][bootstrap_index] = float(np.mean(values))

    alpha = (1.0 - confidence_level) / 2.0
    interval_records = []

    for candidate, reference in comparisons:
        for metric in PERFORMANCE_METRICS:
            values = distributions[(candidate, reference, metric)]
            lower = float(np.quantile(values, alpha))
            upper = float(np.quantile(values, 1.0 - alpha))

            interval_records.append(
                {
                    "candidate_model": candidate,
                    "reference_model": reference,
                    "metric": metric,
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                    "interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                    "bootstrap_resamples": n_resamples,
                    "bootstrap_random_state": random_state,
                    "confidence_level": confidence_level,
                    "resampling_unit": ("exact_predictor_group"),
                }
            )

    intervals = pd.DataFrame.from_records(interval_records)

    return point_estimates.merge(
        intervals,
        on=[
            "candidate_model",
            "reference_model",
            "metric",
        ],
        how="inner",
        validate="one_to_one",
    )


def save_comparisons(
    comparisons: pd.DataFrame,
    *,
    reports_directory: Path = DEFAULT_REPORTS_DIRECTORY,
) -> Path:
    """Save paired model comparisons to CSV."""
    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = reports_directory / COMPARISONS_FILENAME
    comparisons.to_csv(path, index=False)
    return path


def main() -> None:
    """Generate prespecified paired model comparisons."""
    predictions = pd.read_csv(DEFAULT_PREDICTIONS_PATH)
    comparisons = grouped_bootstrap_comparisons(predictions)
    path = save_comparisons(comparisons)

    print("Paired comparison rows:", len(comparisons))
    print(f"comparisons: {path}")


if __name__ == "__main__":
    main()
