"""Tests for the duplicate-removal sensitivity analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import cervical_biopsy_modeling.sensitivity_duplicates as sensitivity
from cervical_biopsy_modeling.evaluation import EvaluationResults
from cervical_biopsy_modeling.modeling_data import ModelingDataset
from cervical_biopsy_modeling.summary import SummaryResults


def make_raw_dataframe() -> pd.DataFrame:
    """Create raw-like data containing one duplicated predictor profile."""

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


def make_deduplicated_modeling_dataset() -> ModelingDataset:
    """Create a minimal modeling dataset with unique groups."""

    return ModelingDataset(
        predictors=pd.DataFrame(
            {
                "Age": [20, 30],
                "risk_feature": [2.0, 1.0],
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
                "outcome": [0, 1],
                "probability": [0.5, 0.5],
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


def test_build_duplicate_sensitivity_dataset_is_outcome_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_data = make_raw_dataframe()

    monkeypatch.setattr(
        sensitivity,
        "load_validated_dataset",
        lambda: raw_data,
    )

    modeling = sensitivity.build_duplicate_sensitivity_dataset()

    assert modeling.predictors["Age"].tolist() == [20, 30]
    assert modeling.outcome.tolist() == [0, 0]
    assert modeling.groups.is_unique
    assert len(modeling.predictors) == 2


def test_run_duplicate_sensitivity_orchestrates_existing_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modeling = make_deduplicated_modeling_dataset()
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
        "build_duplicate_sensitivity_dataset",
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
    monkeypatch.setattr(
        sensitivity,
        "build_model_specifications",
        lambda **kwargs: {"dummy_prior": "specification"},
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
    monkeypatch.setattr(
        sensitivity,
        "build_summary_results",
        lambda predictions, **kwargs: summary,
    )
    monkeypatch.setattr(
        sensitivity,
        "grouped_bootstrap_comparisons",
        lambda predictions, **kwargs: comparisons,
    )

    results = sensitivity.run_duplicate_sensitivity(
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

    assert recorded["split_kwargs"] == {
        "n_splits": 3,
        "n_repeats": 2,
        "random_state": 7,
    }
    assert recorded["evaluation_kwargs"] == {
        "inner_n_splits": 2,
        "threshold": 0.4,
        "n_jobs": 2,
    }


def test_save_duplicate_sensitivity_results_uses_requested_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reports_directory = tmp_path / "sensitivity"
    evaluation = make_evaluation_results()
    summary = make_summary_results()
    comparisons = pd.DataFrame()
    results = sensitivity.DuplicateSensitivityResults(
        modeling=make_deduplicated_modeling_dataset(),
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

    paths = sensitivity.save_duplicate_sensitivity_results(
        results,
        reports_directory=reports_directory,
    )

    assert reports_directory.is_dir()
    assert paths == {
        "fold_metrics": evaluation_path,
        "performance_summary": summary_path,
        "comparisons": comparisons_path,
    }
