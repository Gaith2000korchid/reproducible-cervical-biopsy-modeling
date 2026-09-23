"""Compare primary and no-missing-indicator predictions."""

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
NO_INDICATOR_PREDICTIONS_PATH = Path(
    "reports/sensitivity_no_missing_indicators/outer_predictions.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "reports/sensitivity_no_missing_indicators/paired_missing_indicator_comparisons.csv"
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

MISSING_INDICATOR_COMPARISONS: Final = (
    (
        "logistic_regression_no_missing_indicators",
        "logistic_regression_primary",
    ),
    (
        "xgboost_no_missing_indicators",
        "xgboost_primary",
    ),
)


def build_missing_indicator_prediction_table(
    primary_predictions: pd.DataFrame,
    no_indicator_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align and label primary and no-indicator predictions."""

    required_columns = {
        "model",
        "predicted_probability",
        "predicted_class",
        *ALIGNMENT_COLUMNS,
    }

    for label, predictions in (
        ("primary", primary_predictions),
        ("no-indicator", no_indicator_predictions),
    ):
        missing_columns = required_columns.difference(predictions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"The {label} prediction table is missing: {missing}.")

    labeled_tables: list[pd.DataFrame] = []

    for model in BASE_MODELS:
        primary_model = (
            primary_predictions.loc[primary_predictions["model"] == model]
            .sort_values(list(ALIGNMENT_COLUMNS))
            .reset_index(drop=True)
        )
        no_indicator_model = (
            no_indicator_predictions.loc[no_indicator_predictions["model"] == model]
            .sort_values(list(ALIGNMENT_COLUMNS))
            .reset_index(drop=True)
        )

        if primary_model.empty:
            raise ValueError(f"Primary predictions are missing model {model!r}.")
        if no_indicator_model.empty:
            raise ValueError(f"No-indicator predictions are missing model {model!r}.")

        if not primary_model.loc[
            :,
            ALIGNMENT_COLUMNS,
        ].equals(
            no_indicator_model.loc[
                :,
                ALIGNMENT_COLUMNS,
            ]
        ):
            raise ValueError(
                "Primary and no-indicator predictions are not aligned "
                f"for model {model!r}."
            )

        primary_labeled = primary_model.copy()
        primary_labeled["model"] = f"{model}_primary"

        no_indicator_labeled = no_indicator_model.copy()
        no_indicator_labeled["model"] = f"{model}_no_missing_indicators"

        labeled_tables.extend(
            [
                primary_labeled,
                no_indicator_labeled,
            ]
        )

    return pd.concat(
        labeled_tables,
        ignore_index=True,
    )


def compute_missing_indicator_comparisons(
    primary_predictions: pd.DataFrame,
    no_indicator_predictions: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute paired comparisons for missingness indicators."""

    combined_predictions = build_missing_indicator_prediction_table(
        primary_predictions,
        no_indicator_predictions,
    )

    return grouped_bootstrap_comparisons(
        combined_predictions,
        comparisons=MISSING_INDICATOR_COMPARISONS,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )


def load_and_compute_missing_indicator_comparisons(
    *,
    primary_path: Path = PRIMARY_PREDICTIONS_PATH,
    no_indicator_path: Path = NO_INDICATOR_PREDICTIONS_PATH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Load both prediction tables and compute comparisons."""

    primary_predictions = pd.read_csv(primary_path)
    no_indicator_predictions = pd.read_csv(no_indicator_path)

    return compute_missing_indicator_comparisons(
        primary_predictions,
        no_indicator_predictions,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )


def save_missing_indicator_comparisons(
    comparisons: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save paired missing-indicator comparisons."""

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
    """Compute and save primary-versus-no-indicator comparisons."""

    comparisons = load_and_compute_missing_indicator_comparisons()

    output_path = save_missing_indicator_comparisons(comparisons)

    print(
        "Paired missing-indicator comparison rows:",
        len(comparisons),
    )
    print(
        "missing_indicator_comparisons:",
        output_path,
    )


if __name__ == "__main__":
    main()
