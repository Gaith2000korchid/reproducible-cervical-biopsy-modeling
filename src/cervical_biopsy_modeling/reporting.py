"""Generate final figures from committed result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

DEFAULT_REPORTS_DIRECTORY = Path("reports")
DEFAULT_FIGURES_DIRECTORY = DEFAULT_REPORTS_DIRECTORY / "figures"

DISPLAY_METRICS: Final = (
    "average_precision",
    "roc_auc",
    "brier_score",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
)

METRIC_LABELS: Final = {
    "average_precision": "Average precision",
    "roc_auc": "ROC AUC",
    "brier_score": "Brier score\n(lower is better)",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced accuracy",
}

MODEL_ORDER: Final = (
    "dummy_prior",
    "logistic_regression",
    "xgboost",
)

MODEL_LABELS: Final = {
    "dummy_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "xgboost": "XGBoost",
}

MODEL_COLORS: Final = {
    "dummy_prior": "#6B7280",
    "logistic_regression": "#0072B2",
    "xgboost": "#D55E00",
}

PERFORMANCE_COLUMNS: Final = {
    "model",
    "metric",
    "repeat_mean",
    "bootstrap_lower",
    "bootstrap_upper",
}

COMPARISON_COLUMNS: Final = {
    "candidate_model",
    "reference_model",
    "metric",
    "repeat_mean_difference",
    "bootstrap_lower",
    "bootstrap_upper",
    "difference_definition",
}


def _require_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    *,
    label: str,
) -> None:
    """Require the columns needed for a result figure."""

    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The {label} table is missing: {missing}.")


def _save_figure(
    figure: Figure,
    output_path: Path,
) -> Path:
    """Save and close one figure."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_path


def _metric_performance_rows(
    table: pd.DataFrame,
    metric: str,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Return one performance row per requested model."""

    rows = table.loc[(table["metric"] == metric) & table["model"].isin(models)].copy()

    if rows["model"].duplicated().any():
        raise ValueError(f"Performance table contains duplicate rows for {metric!r}.")

    observed_models = set(rows["model"])
    expected_models = set(models)
    if observed_models != expected_models:
        raise ValueError(
            f"Performance table does not contain every model for {metric!r}."
        )

    return rows.set_index("model").loc[list(models)]


def plot_primary_performance(
    performance_summary: pd.DataFrame,
    *,
    output_path: Path = (DEFAULT_FIGURES_DIRECTORY / "primary_performance.png"),
) -> Path:
    """Plot primary point estimates and grouped-bootstrap intervals."""

    _require_columns(
        performance_summary,
        PERFORMANCE_COLUMNS,
        label="performance summary",
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13, 7.5),
        layout="constrained",
    )

    positions = np.arange(len(MODEL_ORDER))[::-1]

    for axis, metric in zip(
        np.asarray(axes).ravel(),
        DISPLAY_METRICS,
        strict=True,
    ):
        rows = _metric_performance_rows(
            performance_summary,
            metric,
            MODEL_ORDER,
        )

        for position, model in zip(
            positions,
            MODEL_ORDER,
            strict=True,
        ):
            row = rows.loc[model]
            estimate = float(row["repeat_mean"])
            lower = float(row["bootstrap_lower"])
            upper = float(row["bootstrap_upper"])

            axis.errorbar(
                estimate,
                position,
                xerr=np.asarray(
                    [
                        [estimate - lower],
                        [upper - estimate],
                    ]
                ),
                fmt="o",
                color=MODEL_COLORS[model],
                capsize=3,
                markersize=6,
            )

        if metric in {"roc_auc", "balanced_accuracy"}:
            axis.axvline(
                0.5,
                color="#9CA3AF",
                linestyle="--",
                linewidth=1,
            )

        axis.set_xlim(0.0, 1.0)
        axis.set_yticks(
            positions,
            [MODEL_LABELS[model] for model in MODEL_ORDER],
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Estimate")
        axis.grid(
            axis="x",
            alpha=0.25,
        )

    figure.suptitle(
        "Primary nested cross-validation performance",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        (
            "Points are means across five complete outer repetitions; "
            "bars are conditional 95% grouped-bootstrap intervals."
        ),
        ha="center",
        fontsize=9,
    )

    return _save_figure(
        figure,
        output_path,
    )


def plot_duplicate_sensitivity(
    primary_summary: pd.DataFrame,
    deduplicated_summary: pd.DataFrame,
    *,
    output_path: Path = (DEFAULT_FIGURES_DIRECTORY / "duplicate_sensitivity.png"),
) -> Path:
    """Compare primary and deduplicated performance estimates."""

    for label, table in (
        ("primary performance", primary_summary),
        ("deduplicated performance", deduplicated_summary),
    ):
        _require_columns(
            table,
            PERFORMANCE_COLUMNS,
            label=label,
        )

    models = (
        "logistic_regression",
        "xgboost",
    )
    analyses = (
        ("Primary", primary_summary, "o", 0.12),
        ("One row per group", deduplicated_summary, "s", -0.12),
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13, 7.5),
        layout="constrained",
    )

    base_positions = np.arange(len(models))[::-1]

    for axis, metric in zip(
        np.asarray(axes).ravel(),
        DISPLAY_METRICS,
        strict=True,
    ):
        for _, table, marker, offset in analyses:
            rows = _metric_performance_rows(
                table,
                metric,
                models,
            )

            for base_position, model in zip(
                base_positions,
                models,
                strict=True,
            ):
                row = rows.loc[model]
                estimate = float(row["repeat_mean"])
                lower = float(row["bootstrap_lower"])
                upper = float(row["bootstrap_upper"])

                axis.errorbar(
                    estimate,
                    base_position + offset,
                    xerr=np.asarray(
                        [
                            [estimate - lower],
                            [upper - estimate],
                        ]
                    ),
                    fmt=marker,
                    color=MODEL_COLORS[model],
                    capsize=3,
                    markersize=6,
                )

        if metric in {"roc_auc", "balanced_accuracy"}:
            axis.axvline(
                0.5,
                color="#9CA3AF",
                linestyle="--",
                linewidth=1,
            )

        axis.set_xlim(0.0, 1.0)
        axis.set_yticks(
            base_positions,
            [MODEL_LABELS[model] for model in models],
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Estimate")
        axis.grid(
            axis="x",
            alpha=0.25,
        )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#374151",
                linestyle="none",
                label="Primary",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="#374151",
                linestyle="none",
                label="One row per group",
            ),
        ],
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.97),
    )
    figure.suptitle(
        "Duplicate-removal sensitivity analysis",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        (
            "Intervals are conditional grouped-bootstrap intervals. "
            "The two analyses contain different numbers of observations."
        ),
        ha="center",
        fontsize=9,
    )

    return _save_figure(
        figure,
        output_path,
    )


def _plot_oriented_comparisons(
    comparisons: pd.DataFrame,
    *,
    label_order: tuple[str, ...],
    colors: dict[str, str],
    title: str,
    note: str,
    output_path: Path,
) -> Path:
    """Plot comparison intervals where positive favors the candidate."""

    _require_columns(
        comparisons,
        COMPARISON_COLUMNS | {"display_label"},
        label="comparison",
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        layout="constrained",
    )

    positions = np.arange(len(label_order))[::-1]

    for axis, metric in zip(
        np.asarray(axes).ravel(),
        DISPLAY_METRICS,
        strict=True,
    ):
        rows = comparisons.loc[comparisons["metric"] == metric].copy()

        if rows["display_label"].duplicated().any():
            raise ValueError(
                f"Comparison table contains duplicate rows for {metric!r}."
            )

        observed_labels = set(rows["display_label"])
        expected_labels = set(label_order)
        if observed_labels != expected_labels:
            raise ValueError(
                f"Comparison table does not contain every label for {metric!r}."
            )

        rows = rows.set_index("display_label").loc[list(label_order)]

        interval_limit = 0.0
        for position, label in zip(
            positions,
            label_order,
            strict=True,
        ):
            row = rows.loc[label]
            estimate = float(row["repeat_mean_difference"])
            lower = float(row["bootstrap_lower"])
            upper = float(row["bootstrap_upper"])
            interval_limit = max(
                interval_limit,
                abs(lower),
                abs(upper),
            )

            axis.errorbar(
                estimate,
                position,
                xerr=np.asarray(
                    [
                        [estimate - lower],
                        [upper - estimate],
                    ]
                ),
                fmt="o",
                color=colors[label],
                capsize=3,
                markersize=6,
            )

        limit = max(0.05, interval_limit * 1.15)
        axis.axvline(
            0.0,
            color="#111827",
            linestyle="--",
            linewidth=1,
        )
        axis.set_xlim(-limit, limit)
        axis.set_yticks(
            positions,
            label_order,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("Oriented difference")
        axis.grid(
            axis="x",
            alpha=0.25,
        )

    figure.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        note,
        ha="center",
        fontsize=9,
    )

    return _save_figure(
        figure,
        output_path,
    )


def plot_modeling_sensitivities(
    weighting_comparisons: pd.DataFrame,
    missing_indicator_comparisons: pd.DataFrame,
    *,
    output_path: Path = (
        DEFAULT_FIGURES_DIRECTORY / "modeling_sensitivity_comparisons.png"
    ),
) -> Path:
    """Plot weighting and missing-indicator comparisons."""

    weighting = weighting_comparisons.copy()
    weighting["display_label"] = weighting["candidate_model"].map(
        {
            "logistic_regression_weighted": "Logistic: weighted",
            "xgboost_weighted": "XGBoost: weighted",
        }
    )

    missing_indicators = missing_indicator_comparisons.copy()
    missing_indicators["display_label"] = missing_indicators["candidate_model"].map(
        {
            "logistic_regression_no_missing_indicators": ("Logistic: no indicators"),
            "xgboost_no_missing_indicators": "XGBoost: no indicators",
        }
    )

    combined = pd.concat(
        [
            weighting,
            missing_indicators,
        ],
        ignore_index=True,
    )

    label_order = (
        "Logistic: weighted",
        "XGBoost: weighted",
        "Logistic: no indicators",
        "XGBoost: no indicators",
    )
    colors = {
        "Logistic: weighted": MODEL_COLORS["logistic_regression"],
        "XGBoost: weighted": MODEL_COLORS["xgboost"],
        "Logistic: no indicators": MODEL_COLORS["logistic_regression"],
        "XGBoost: no indicators": MODEL_COLORS["xgboost"],
    }

    return _plot_oriented_comparisons(
        combined,
        label_order=label_order,
        colors=colors,
        title="Modeling-choice sensitivity analyses",
        note=(
            "Positive values favor the named candidate. "
            "Weighting is compared with no weighting; no indicators "
            "is compared with the primary preprocessing."
        ),
        output_path=output_path,
    )


def plot_diagnostic_leakage_audit(
    diagnostic_comparisons: pd.DataFrame,
    *,
    output_path: Path = (DEFAULT_FIGURES_DIRECTORY / "diagnostic_leakage_audit.png"),
) -> Path:
    """Plot diagnostic-inclusive versus primary comparisons."""

    comparisons = diagnostic_comparisons.copy()
    comparisons["display_label"] = comparisons["candidate_model"].map(
        {
            "logistic_regression_diagnostic_inclusive": (
                "Logistic: diagnostic-inclusive"
            ),
            "xgboost_diagnostic_inclusive": ("XGBoost: diagnostic-inclusive"),
        }
    )

    label_order = (
        "Logistic: diagnostic-inclusive",
        "XGBoost: diagnostic-inclusive",
    )
    colors = {
        "Logistic: diagnostic-inclusive": MODEL_COLORS["logistic_regression"],
        "XGBoost: diagnostic-inclusive": MODEL_COLORS["xgboost"],
    }

    return _plot_oriented_comparisons(
        comparisons,
        label_order=label_order,
        colors=colors,
        title="Diagnostic-variable leakage audit",
        note=(
            "Positive values favor diagnostic-inclusive predictors. "
            "These results are intentionally leakage-prone and are not "
            "valid pre-screening performance estimates."
        ),
        output_path=output_path,
    )


def generate_final_figures(
    *,
    reports_directory: Path = DEFAULT_REPORTS_DIRECTORY,
    figures_directory: Path = DEFAULT_FIGURES_DIRECTORY,
) -> dict[str, Path]:
    """Load committed tables and generate all final figures."""

    primary_summary = pd.read_csv(reports_directory / "performance_summary.csv")
    primary_comparisons = pd.read_csv(
        reports_directory / "paired_model_comparisons.csv"
    )
    deduplicated_summary = pd.read_csv(
        reports_directory / "sensitivity_duplicates" / "performance_summary.csv"
    )
    weighting_comparisons = pd.read_csv(
        reports_directory
        / "sensitivity_unweighted"
        / "paired_weighting_comparisons.csv"
    )
    diagnostic_comparisons = pd.read_csv(
        reports_directory
        / "sensitivity_diagnostic_inclusive"
        / "paired_diagnostic_comparisons.csv"
    )
    missing_indicator_comparisons = pd.read_csv(
        reports_directory
        / "sensitivity_no_missing_indicators"
        / "paired_missing_indicator_comparisons.csv"
    )

    del primary_comparisons

    return {
        "primary_performance": plot_primary_performance(
            primary_summary,
            output_path=figures_directory / "primary_performance.png",
        ),
        "duplicate_sensitivity": plot_duplicate_sensitivity(
            primary_summary,
            deduplicated_summary,
            output_path=(figures_directory / "duplicate_sensitivity.png"),
        ),
        "modeling_sensitivities": plot_modeling_sensitivities(
            weighting_comparisons,
            missing_indicator_comparisons,
            output_path=(figures_directory / "modeling_sensitivity_comparisons.png"),
        ),
        "diagnostic_leakage_audit": plot_diagnostic_leakage_audit(
            diagnostic_comparisons,
            output_path=(figures_directory / "diagnostic_leakage_audit.png"),
        ),
    }


def main() -> None:
    """Generate and report all final figure paths."""

    paths = generate_final_figures()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
