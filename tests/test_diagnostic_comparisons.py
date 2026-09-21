"""Tests for primary-versus-diagnostic comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import cervical_biopsy_modeling.diagnostic_comparisons as diagnostic


def make_predictions(
    *,
    probability_shift: float = 0.0,
) -> pd.DataFrame:
    """Create aligned predictions for both fitted models."""

    rows: list[dict[str, object]] = []

    for model in (
        "logistic_regression",
        "xgboost",
    ):
        probabilities = [
            0.10 + probability_shift,
            0.70 + probability_shift,
            0.20 + probability_shift,
            0.80 + probability_shift,
        ]

        for row_position, (
            fold,
            group,
            outcome,
            probability,
        ) in enumerate(
            zip(
                [1, 1, 2, 2],
                [10, 11, 12, 13],
                [0, 1, 0, 1],
                probabilities,
                strict=True,
            )
        ):
            rows.append(
                {
                    "repeat": 1,
                    "fold": fold,
                    "random_state": 42,
                    "model": model,
                    "row_position": row_position,
                    "group": group,
                    "observed_outcome": outcome,
                    "predicted_probability": probability,
                    "predicted_class": int(probability >= 0.5),
                    "threshold": 0.5,
                }
            )

    return pd.DataFrame(rows)


def test_build_diagnostic_prediction_table_labels_four_variants() -> None:
    """Both predictor modes must receive distinct model labels."""

    primary = make_predictions()

    diagnostic_predictions = make_predictions(
        probability_shift=0.05,
    )

    combined = diagnostic.build_diagnostic_comparison_prediction_table(
        primary,
        diagnostic_predictions,
    )

    assert len(combined) == 16

    assert set(combined["model"]) == {
        "logistic_regression_primary",
        "logistic_regression_diagnostic_inclusive",
        "xgboost_primary",
        "xgboost_diagnostic_inclusive",
    }

    counts = combined.groupby("model").size().to_dict()

    assert counts == {
        "logistic_regression_primary": 4,
        "logistic_regression_diagnostic_inclusive": 4,
        "xgboost_primary": 4,
        "xgboost_diagnostic_inclusive": 4,
    }


def test_build_diagnostic_prediction_table_preserves_probabilities() -> None:
    """Prediction probabilities must not be altered during labeling."""

    primary = make_predictions()

    diagnostic_predictions = make_predictions(
        probability_shift=0.05,
    )

    combined = diagnostic.build_diagnostic_comparison_prediction_table(
        primary,
        diagnostic_predictions,
    )

    primary_logistic = combined.loc[
        combined["model"] == "logistic_regression_primary",
        "predicted_probability",
    ].reset_index(drop=True)

    diagnostic_logistic = combined.loc[
        combined["model"] == "logistic_regression_diagnostic_inclusive",
        "predicted_probability",
    ].reset_index(drop=True)

    assert primary_logistic.tolist() == pytest.approx(
        [
            0.10,
            0.70,
            0.20,
            0.80,
        ]
    )

    assert diagnostic_logistic.tolist() == pytest.approx(
        [
            0.15,
            0.75,
            0.25,
            0.85,
        ]
    )


def test_build_diagnostic_prediction_table_rejects_misalignment() -> None:
    """Different outer-fold membership must stop the comparison."""

    primary = make_predictions()
    diagnostic_predictions = make_predictions()

    mismatch = (diagnostic_predictions["model"] == "xgboost") & (
        diagnostic_predictions["row_position"] == 0
    )

    diagnostic_predictions.loc[mismatch, "fold"] = 99

    with pytest.raises(
        ValueError,
        match="not aligned",
    ):
        diagnostic.build_diagnostic_comparison_prediction_table(
            primary,
            diagnostic_predictions,
        )


def test_build_diagnostic_prediction_table_rejects_missing_columns() -> None:
    """Incomplete prediction tables must raise a clear error."""

    primary = make_predictions().drop(columns="predicted_probability")

    diagnostic_predictions = make_predictions()

    with pytest.raises(
        ValueError,
        match="predicted_probability",
    ):
        diagnostic.build_diagnostic_comparison_prediction_table(
            primary,
            diagnostic_predictions,
        )


def test_compute_diagnostic_comparisons_uses_paired_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grouped bootstrap must compare matching model families."""

    primary = make_predictions()

    diagnostic_predictions = make_predictions(
        probability_shift=0.05,
    )

    expected = pd.DataFrame(
        {
            "metric": ["average_precision"],
        }
    )

    recorded: dict[str, object] = {}

    def fake_grouped_bootstrap(
        predictions: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        recorded["predictions"] = predictions
        recorded["kwargs"] = kwargs

        return expected

    monkeypatch.setattr(
        diagnostic,
        "grouped_bootstrap_comparisons",
        fake_grouped_bootstrap,
    )

    result = diagnostic.compute_diagnostic_comparisons(
        primary,
        diagnostic_predictions,
        n_resamples=10,
        random_state=7,
        confidence_level=0.90,
    )

    assert result is expected

    combined = recorded["predictions"]

    assert isinstance(combined, pd.DataFrame)

    assert set(combined["model"]) == {
        "logistic_regression_primary",
        "logistic_regression_diagnostic_inclusive",
        "xgboost_primary",
        "xgboost_diagnostic_inclusive",
    }

    assert recorded["kwargs"] == {
        "comparisons": diagnostic.DIAGNOSTIC_COMPARISONS,
        "n_resamples": 10,
        "random_state": 7,
        "confidence_level": 0.90,
    }


def test_save_diagnostic_comparisons_writes_requested_file(
    tmp_path: Path,
) -> None:
    """The result table must be written without an index column."""

    comparisons = pd.DataFrame(
        {
            "candidate_model": ["xgboost_diagnostic_inclusive"],
            "reference_model": ["xgboost_primary"],
            "metric": ["average_precision"],
        }
    )

    output_path = tmp_path / "nested" / "comparisons.csv"

    returned_path = diagnostic.save_diagnostic_comparisons(
        comparisons,
        output_path=output_path,
    )

    assert returned_path == output_path
    assert output_path.is_file()

    saved = pd.read_csv(output_path)

    pd.testing.assert_frame_equal(
        saved,
        comparisons,
    )
