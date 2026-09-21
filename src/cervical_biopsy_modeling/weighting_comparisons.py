"""Compare weighted and unweighted model predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from cervical_biopsy_modeling.comparisons import (
    grouped_bootstrap_comparisons,
)

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_RESAMPLES = 2000

PRIMARY_PREDICTIONS_PATH = Path("reports/outer_predictions.csv")
UNWEIGHTED_PREDICTIONS_PATH = Path(
    "reports/sensitivity_unweighted/outer_predictions.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "reports/sensitivity_unweighted/paired_weighting_comparisons.csv"
)

BASE_MODELS: Final = (
    "logistic_regression",
    "xgboost",
)

ALIGNMENT_COLUMNS: Final = (
    "repeat",
    "fold",
    "random_state",
    "row_position",
    "group",
    "observed_outcome",
    "threshold",
)

WEIGHTING_COMPARISONS: Final = (
    (
        "logistic_regression_weighted",
        "logistic_regression_unweighted",
    ),
    (
        "xgboost_weighted",
        "xgboost_unweighted",
    ),
)


def build_weighting_prediction_table(
    weighted_predictions: pd.DataFrame,
    unweighted_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align and label weighted and unweighted predictions."""

    required_columns = {
        "model",
        "predicted_probability",
        "predicted_class",
        *ALIGNMENT_COLUMNS,
    }

    for label, predictions in (
        ("weighted", weighted_predictions),
        ("unweighted", unweighted_predictions),
    ):
        missing_columns = required_columns.difference(predictions.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"The {label} prediction table is missing: {missing}.")

    labeled_tables: list[pd.DataFrame] = []

    for model in BASE_MODELS:
        weighted_model = (
            weighted_predictions.loc[weighted_predictions["model"] == model]
            .sort_values(list(ALIGNMENT_COLUMNS))
            .reset_index(drop=True)
        )

        unweighted_model = (
            unweighted_predictions.loc[unweighted_predictions["model"] == model]
            .sort_values(list(ALIGNMENT_COLUMNS))
            .reset_index(drop=True)
        )

        if weighted_model.empty:
            raise ValueError(f"Weighted predictions are missing model {model!r}.")

        if unweighted_model.empty:
            raise ValueError(f"Unweighted predictions are missing model {model!r}.")

        if not weighted_model.loc[
            :,
            ALIGNMENT_COLUMNS,
        ].equals(
            unweighted_model.loc[
                :,
                ALIGNMENT_COLUMNS,
            ]
        ):
            raise ValueError(
                f"Weighted and unweighted predictions are not aligned "
                f"for model {model!r}."
            )

        weighted_labeled = weighted_model.copy()
        weighted_labeled["model"] = f"{model}_weighted"

        unweighted_labeled = unweighted_model.copy()
        unweighted_labeled["model"] = f"{model}_unweighted"

        labeled_tables.extend(
            [
                weighted_labeled,
                unweighted_labeled,
            ]
        )

    return pd.concat(
        labeled_tables,
        ignore_index=True,
    )


def compute_weighting_comparisons(
    weighted_predictions: pd.DataFrame,
    unweighted_predictions: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute paired grouped comparisons for class weighting."""

    combined_predictions = build_weighting_prediction_table(
        weighted_predictions,
        unweighted_predictions,
    )

    return grouped_bootstrap_comparisons(
        combined_predictions,
        comparisons=WEIGHTING_COMPARISONS,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )


def load_and_compute_weighting_comparisons(
    *,
    weighted_path: Path = PRIMARY_PREDICTIONS_PATH,
    unweighted_path: Path = UNWEIGHTED_PREDICTIONS_PATH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Load both prediction tables and compute comparisons."""

    weighted_predictions = pd.read_csv(weighted_path)
    unweighted_predictions = pd.read_csv(unweighted_path)

    return compute_weighting_comparisons(
        weighted_predictions,
        unweighted_predictions,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )


def save_weighting_comparisons(
    comparisons: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save paired weighting comparisons."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparisons.to_csv(
        output_path,
        index=False,
    )

    return output_path


def main() -> None:
    """Compute and save weighted-versus-unweighted comparisons."""

    comparisons = load_and_compute_weighting_comparisons()
    output_path = save_weighting_comparisons(comparisons)

    print("Paired weighting comparison rows:", len(comparisons))
    print("weighting_comparisons:", output_path)


if __name__ == "__main__":
    main()
