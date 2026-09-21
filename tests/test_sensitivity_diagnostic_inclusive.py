"""Tests for the diagnostic-inclusive leakage audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import cervical_biopsy_modeling.sensitivity_diagnostic_inclusive as sensitivity
from cervical_biopsy_modeling.audit import LEAKAGE_COLUMNS
from cervical_biopsy_modeling.evaluation import EvaluationResults
from cervical_biopsy_modeling.modeling_data import ModelingDataset
from cervical_biopsy_modeling.summary import SummaryResults


def make_raw_dataframe() -> pd.DataFrame:
    """Create raw-like data containing diagnostic variables."""

    return pd.DataFrame(
        {
            "Age": [20, 20, 30],
            "risk_feature": [2.0, 2.0, 1.0],
            "Hinselmann": [0, 1, 0],
            "Schiller": [0, 1, 0],
            "Citology": [0, 1, 0],
            "Dx:Cancer": [0, 0, 0],
            "Dx:CIN": [0, 0, 0],
            "Dx:HPV": [0, 0, 0],
            "Dx": [0, 0, 0],
            "Biopsy": [0, 1, 0],
        }
    )


def make_diagnostic_modeling_dataset() -> ModelingDataset:
    """Create a minimal diagnostic-inclusive modeling dataset."""

    return ModelingDataset(
        predictors=pd.DataFrame(
            {
                "Age": [20, 30],
                "risk_feature": [2.0, 1.0],
                "Hinselmann": [0, 1],
            }
        ),
        outcome=pd.Series(
            [0, 1],
            name="Biopsy",
            dtype="int8",
        ),
        groups=pd.Series(
            [0, 1],
            name="predictor_group",
        ),
    )


def make_evaluation_results() -> EvaluationResults:
    """Create minimal evaluation tables."""

    return EvaluationResults(
        fold_metrics=pd.DataFrame(
            {
                "repeat": [1],
                "fold": [1],
                "model": ["dummy_prior"],
            }
        ),
        predictions=pd.DataFrame(
            {
                "repeat": [1, 1],
                "fold": [1, 1],
                "model": ["dummy_prior", "dummy_prior"],
                "group": [0, 1],
                "observed_outcome": [0, 1],
                "predicted_probability": [0.5, 0.5],
            }
        ),
        selected_parameters=pd.DataFrame(),
        inner_candidates=pd.DataFrame(),
    )


def make_summary_results() -> SummaryResults:
    """Create minimal summary tables."""

    return SummaryResults(
        repeat_metrics=pd.DataFrame(
            {
                "model": ["dummy_prior"],
                "repeat": [1],
            }
        ),
        performance_summary=pd.DataFrame(
            {
                "model": ["dummy_prior"],
                "metric": ["average_precision"],
            }
        ),
    )


def test_build_diagnostic_inclusive_audit_dataset_retains_leakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_data = make_raw_dataframe()

    monkeypatch.setattr(
        sensitivity,
        "load_validated_dataset",
        lambda: raw_data,
    )

    modeling = sensitivity.build_diagnostic_inclusive_audit_dataset()

    expected_columns = [column for column in raw_data.columns if column != "Biopsy"]

    assert modeling.predictors.columns.tolist() == expected_columns
    assert set(LEAKAGE_COLUMNS).issubset(modeling.predictors.columns)
    assert modeling.outcome.tolist() == [0, 1, 0]

    # Rows zero and one differ diagnostically but share the same
    # prespecified primary predictor profile.
    assert modeling.groups.iloc[0] == modeling.groups.iloc[1]
    assert modeling.groups.iloc[0] != modeling.groups.iloc[2]


def test_run_diagnostic_inclusive_audit_orchestrates_existing_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling = make_diagnostic_modeling_dataset()
    evaluation = make_evaluation_results()
    summary = make_summary_results()

    comparisons = pd.DataFrame(
        {
            "candidate_model": ["logistic_regression"],
            "reference_model": ["dummy_prior"],
        }
    )

    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        sensitivity,
        "build_diagnostic_inclusive_audit_dataset",
        lambda: modeling,
    )

    def fake_build_splits(
        predictors: pd.DataFrame,
        outcome: pd.Series,
        groups: pd.Series,
        **kwargs: object,
    ) -> list[str]:
        recorded["split_inputs"] = (
            predictors,
            outcome,
            groups,
        )
        recorded["split_kwargs"] = kwargs
        return ["outer-split"]

    monkeypatch.setattr(
        sensitivity,
        "build_repeated_group_splits",
        fake_build_splits,
    )

    def fake_build_specifications(
        **kwargs: object,
    ) -> dict[str, str]:
        recorded["specification_kwargs"] = kwargs
        return {"dummy_prior": "specification"}

    monkeypatch.setattr(
        sensitivity,
        "build_model_specifications",
        fake_build_specifications,
    )

    def fake_evaluate(
        received_modeling: ModelingDataset,
        outer_splits: list[str],
        specifications: dict[str, str],
        **kwargs: object,
    ) -> EvaluationResults:
        recorded["evaluation_inputs"] = (
            received_modeling,
            outer_splits,
            specifications,
        )
        recorded["evaluation_kwargs"] = kwargs
        return evaluation

    monkeypatch.setattr(
        sensitivity,
        "evaluate_outer_splits",
        fake_evaluate,
    )

    def fake_build_summary(
        predictions: pd.DataFrame,
        **kwargs: object,
    ) -> SummaryResults:
        recorded["summary_predictions"] = predictions
        recorded["summary_kwargs"] = kwargs
        return summary

    monkeypatch.setattr(
        sensitivity,
        "build_summary_results",
        fake_build_summary,
    )

    def fake_build_comparisons(
        predictions: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        recorded["comparison_predictions"] = predictions
        recorded["comparison_kwargs"] = kwargs
        return comparisons

    monkeypatch.setattr(
        sensitivity,
        "grouped_bootstrap_comparisons",
        fake_build_comparisons,
    )

    results = sensitivity.run_diagnostic_inclusive_audit(
        outer_n_splits=3,
        outer_n_repeats=2,
        inner_n_splits=2,
        n_resamples=10,
        random_state=7,
        threshold=0.4,
        n_jobs=2,
    )

    assert results.modeling is modeling
    assert results.evaluation is evaluation
    assert results.summary is summary
    assert results.comparisons is comparisons

    split_inputs = recorded["split_inputs"]
    assert isinstance(split_inputs, tuple)
    assert split_inputs[0] is modeling.predictors
    assert split_inputs[1] is modeling.outcome
    assert split_inputs[2] is modeling.groups

    assert recorded["split_kwargs"] == {
        "n_splits": 3,
        "n_repeats": 2,
        "random_state": 7,
    }

    assert recorded["specification_kwargs"] == {
        "random_state": 7,
    }

    assert recorded["evaluation_kwargs"] == {
        "inner_n_splits": 2,
        "threshold": 0.4,
        "n_jobs": 2,
    }

    assert recorded["summary_predictions"] is evaluation.predictions
    assert recorded["summary_kwargs"] == {
        "n_resamples": 10,
        "random_state": 7,
    }

    assert recorded["comparison_predictions"] is evaluation.predictions
    assert recorded["comparison_kwargs"] == {
        "n_resamples": 10,
        "random_state": 7,
    }


def test_save_diagnostic_inclusive_audit_results_uses_requested_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reports_directory = tmp_path / "diagnostic_audit"

    evaluation = make_evaluation_results()
    summary = make_summary_results()
    comparisons = pd.DataFrame()

    results = sensitivity.DiagnosticInclusiveAuditResults(
        modeling=make_diagnostic_modeling_dataset(),
        evaluation=evaluation,
        summary=summary,
        comparisons=comparisons,
    )

    evaluation_path = reports_directory / "outer_fold_metrics.csv"
    summary_path = reports_directory / "performance_summary.csv"
    comparisons_path = reports_directory / "paired_model_comparisons.csv"

    monkeypatch.setattr(
        sensitivity,
        "save_evaluation_results",
        lambda results, *, reports_directory: {
            "fold_metrics": evaluation_path,
        },
    )

    monkeypatch.setattr(
        sensitivity,
        "save_summary_results",
        lambda results, *, reports_directory: {
            "performance_summary": summary_path,
        },
    )

    monkeypatch.setattr(
        sensitivity,
        "save_comparisons",
        lambda comparisons, *, reports_directory: comparisons_path,
    )

    paths = sensitivity.save_diagnostic_inclusive_audit_results(
        results,
        reports_directory=reports_directory,
    )

    assert reports_directory.is_dir()

    assert paths == {
        "fold_metrics": evaluation_path,
        "performance_summary": summary_path,
        "comparisons": comparisons_path,
    }
