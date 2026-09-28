"""Tests for final result figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

import cervical_biopsy_modeling.reporting as reporting  # noqa: E402


def make_performance_summary(
    *,
    shift: float = 0.0,
) -> pd.DataFrame:
    """Create a complete synthetic performance summary."""

    records: list[dict[str, object]] = []

    for model_index, model in enumerate(reporting.MODEL_ORDER):
        for metric_index, metric in enumerate(reporting.DISPLAY_METRICS):
            estimate = 0.20 + 0.10 * model_index + 0.01 * metric_index + shift
            records.append(
                {
                    "model": model,
                    "metric": metric,
                    "repeat_mean": estimate,
                    "bootstrap_lower": estimate - 0.03,
                    "bootstrap_upper": estimate + 0.03,
                }
            )

    return pd.DataFrame(records)


def make_comparisons(
    candidates: tuple[str, ...],
) -> pd.DataFrame:
    """Create complete synthetic oriented comparisons."""

    records: list[dict[str, object]] = []

    for candidate_index, candidate in enumerate(candidates):
        for metric_index, metric in enumerate(reporting.DISPLAY_METRICS):
            estimate = 0.01 + 0.01 * candidate_index + 0.001 * metric_index
            difference_definition = (
                "reference_minus_candidate"
                if metric == "brier_score"
                else "candidate_minus_reference"
            )
            records.append(
                {
                    "candidate_model": candidate,
                    "reference_model": "reference",
                    "metric": metric,
                    "repeat_mean_difference": estimate,
                    "bootstrap_lower": estimate - 0.02,
                    "bootstrap_upper": estimate + 0.02,
                    "difference_definition": (difference_definition),
                }
            )

    return pd.DataFrame(records)


def fake_save_figure(
    figure: Figure,
    output_path: Path,
) -> Path:
    """Replace expensive image rendering in orchestration tests."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_bytes(b"synthetic figure")
    figure.clear()
    return output_path


def test_plot_primary_performance_writes_png(
    tmp_path: Path,
) -> None:
    """The primary figure must be saved at the requested path."""

    output_path = tmp_path / "primary.png"

    returned_path = reporting.plot_primary_performance(
        make_performance_summary(),
        output_path=output_path,
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_primary_performance_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    """Incomplete summaries must raise a clear error."""

    summary = make_performance_summary().drop(columns="bootstrap_upper")

    with pytest.raises(
        ValueError,
        match="bootstrap_upper",
    ):
        reporting.plot_primary_performance(
            summary,
            output_path=tmp_path / "primary.png",
        )


def test_plot_duplicate_sensitivity_accepts_complete_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Primary and deduplicated estimates must be plottable."""

    monkeypatch.setattr(
        reporting,
        "_save_figure",
        fake_save_figure,
    )
    output_path = tmp_path / "duplicates.png"

    returned_path = reporting.plot_duplicate_sensitivity(
        make_performance_summary(),
        make_performance_summary(shift=0.01),
        output_path=output_path,
    )

    assert returned_path == output_path
    assert output_path.read_bytes() == b"synthetic figure"


def test_plot_modeling_sensitivities_accepts_complete_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Weighting and missing-indicator comparisons must align."""

    monkeypatch.setattr(
        reporting,
        "_save_figure",
        fake_save_figure,
    )

    weighting = make_comparisons(
        (
            "logistic_regression_weighted",
            "xgboost_weighted",
        )
    )
    missing_indicators = make_comparisons(
        (
            "logistic_regression_no_missing_indicators",
            "xgboost_no_missing_indicators",
        )
    )
    output_path = tmp_path / "modeling_sensitivities.png"

    returned_path = reporting.plot_modeling_sensitivities(
        weighting,
        missing_indicators,
        output_path=output_path,
    )

    assert returned_path == output_path
    assert output_path.read_bytes() == b"synthetic figure"


def test_plot_diagnostic_audit_rejects_incomplete_models(
    tmp_path: Path,
) -> None:
    """Both diagnostic-inclusive model families are required."""

    comparisons = make_comparisons(("logistic_regression_diagnostic_inclusive",))

    with pytest.raises(
        ValueError,
        match="every label",
    ):
        reporting.plot_diagnostic_leakage_audit(
            comparisons,
            output_path=tmp_path / "diagnostic.png",
        )


def test_generate_final_figures_uses_requested_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The complete figure workflow must use committed input tables."""

    reports_directory = tmp_path / "reports"
    figures_directory = tmp_path / "figures"

    reports_directory.mkdir()
    (reports_directory / "sensitivity_duplicates").mkdir()
    (reports_directory / "sensitivity_unweighted").mkdir()
    (reports_directory / "sensitivity_diagnostic_inclusive").mkdir()
    (reports_directory / "sensitivity_no_missing_indicators").mkdir()

    primary_summary = make_performance_summary()
    primary_summary.to_csv(
        reports_directory / "performance_summary.csv",
        index=False,
    )
    make_comparisons(("logistic_regression",)).to_csv(
        reports_directory / "paired_model_comparisons.csv",
        index=False,
    )
    make_performance_summary(shift=0.01).to_csv(
        reports_directory / "sensitivity_duplicates" / "performance_summary.csv",
        index=False,
    )
    make_comparisons(
        (
            "logistic_regression_weighted",
            "xgboost_weighted",
        )
    ).to_csv(
        reports_directory
        / "sensitivity_unweighted"
        / "paired_weighting_comparisons.csv",
        index=False,
    )
    make_comparisons(
        (
            "logistic_regression_diagnostic_inclusive",
            "xgboost_diagnostic_inclusive",
        )
    ).to_csv(
        reports_directory
        / "sensitivity_diagnostic_inclusive"
        / "paired_diagnostic_comparisons.csv",
        index=False,
    )
    make_comparisons(
        (
            "logistic_regression_no_missing_indicators",
            "xgboost_no_missing_indicators",
        )
    ).to_csv(
        reports_directory
        / "sensitivity_no_missing_indicators"
        / "paired_missing_indicator_comparisons.csv",
        index=False,
    )

    monkeypatch.setattr(
        reporting,
        "_save_figure",
        fake_save_figure,
    )

    paths = reporting.generate_final_figures(
        reports_directory=reports_directory,
        figures_directory=figures_directory,
    )

    assert set(paths) == {
        "primary_performance",
        "duplicate_sensitivity",
        "modeling_sensitivities",
        "diagnostic_leakage_audit",
    }
    assert all(path.is_file() for path in paths.values())
    assert all(path.parent == figures_directory for path in paths.values())
