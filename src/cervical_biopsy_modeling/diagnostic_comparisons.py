"""Compare primary and diagnostic-inclusive model predictions."""

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

DIAGNOSTIC_PREDICTIONS_PATH = Path(
    "reports/sensitivity_diagnostic_inclusive/outer_predictions.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "reports/sensitivity_diagnostic_inclusive/paired_diagnostic_comparisons.csv"
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

DIAGNOSTIC_COMPARISONS: Final = (
    (
        "logistic_regression_diagnostic_inclusive",
        "logistic_regression_primary",
    ),
    (
        "xgboost_diagnostic_inclusive",
        "xgboost_primary",
    ),
)


def build_diagnostic_comparison_prediction_table(
    primary_predictions: pd.DataFrame,
    diagnostic_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Align and label primary and diagnostic-inclusive predictions."""

    required_columns = {
        "model",
        "predicted_probability",
        "predicted_class",
        *ALIGNMENT_COLUMNS,
    }

    for label, predictions in (
        ("primary", primary_predictions),
        ("diagnostic-inclusive", diagnostic_predictions),
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

        diagnostic_model = (
            diagnostic_predictions.loc[diagnostic_predictions["model"] == model]
            .sort_values(list(ALIGNMENT_COLUMNS))
            .reset_index(drop=True)
        )

        if primary_model.empty:
            raise ValueError(f"Primary predictions are missing model {model!r}.")

        if diagnostic_model.empty:
            raise ValueError(
                f"Diagnostic-inclusive predictions are missing model {model!r}."
            )

        if not primary_model.loc[
            :,
            ALIGNMENT_COLUMNS,
        ].equals(
            diagnostic_model.loc[
                :,
                ALIGNMENT_COLUMNS,
            ]
        ):
            raise ValueError(
                "Primary and diagnostic-inclusive predictions "
                f"are not aligned for model {model!r}."
            )

        primary_labeled = primary_model.copy()
        primary_labeled["model"] = f"{model}_primary"

        diagnostic_labeled = diagnostic_model.copy()
        diagnostic_labeled["model"] = f"{model}_diagnostic_inclusive"

        labeled_tables.extend(
            [
                primary_labeled,
                diagnostic_labeled,
            ]
        )

    return pd.concat(
        labeled_tables,
        ignore_index=True,
    )


def compute_diagnostic_comparisons(
    primary_predictions: pd.DataFrame,
    diagnostic_predictions: pd.DataFrame,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute paired comparisons for diagnostic-variable inclusion."""

    combined_predictions = build_diagnostic_comparison_prediction_table(
        primary_predictions,
        diagnostic_predictions,
    )

    return grouped_bootstrap_comparisons(
        combined_predictions,
        comparisons=DIAGNOSTIC_COMPARISONS,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )


def load_and_compute_diagnostic_comparisons(
    *,
    primary_path: Path = PRIMARY_PREDICTIONS_PATH,
    diagnostic_path: Path = DIAGNOSTIC_PREDICTIONS_PATH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Load both prediction tables and compute comparisons."""

    primary_predictions = pd.read_csv(primary_path)
    diagnostic_predictions = pd.read_csv(diagnostic_path)

    return compute_diagnostic_comparisons(
        primary_predictions,
        diagnostic_predictions,
        n_resamples=n_resamples,
        random_state=random_state,
        confidence_level=confidence_level,
    )


def save_diagnostic_comparisons(
    comparisons: pd.DataFrame,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save paired diagnostic-inclusive comparisons."""

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
    """Compute and save primary-versus-diagnostic comparisons."""

    comparisons = load_and_compute_diagnostic_comparisons()

    output_path = save_diagnostic_comparisons(comparisons)

    print(
        "Paired diagnostic comparison rows:",
        len(comparisons),
    )

    print(
        "diagnostic_comparisons:",
        output_path,
    )


if __name__ == "__main__":
    main()
