"""Run the unweighted-model sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cervical_biopsy_modeling.audit import load_validated_dataset
from cervical_biopsy_modeling.comparisons import (
    grouped_bootstrap_comparisons,
    save_comparisons,
)
from cervical_biopsy_modeling.evaluation import (
    EvaluationResults,
    evaluate_outer_splits,
    save_evaluation_results,
)
from cervical_biopsy_modeling.modeling_data import (
    ModelingDataset,
    build_modeling_dataset,
)
from cervical_biopsy_modeling.models import build_model_specifications
from cervical_biopsy_modeling.summary import (
    SummaryResults,
    build_summary_results,
    save_summary_results,
)
from cervical_biopsy_modeling.validation import build_repeated_group_splits

DEFAULT_RANDOM_STATE = 42
DEFAULT_REPORTS_DIRECTORY = Path("reports/sensitivity_unweighted")


@dataclass(frozen=True)
class UnweightedSensitivityResults:
    """All results produced by the unweighted sensitivity analysis."""

    modeling: ModelingDataset
    evaluation: EvaluationResults
    summary: SummaryResults
    comparisons: pd.DataFrame


def build_unweighted_sensitivity_dataset() -> ModelingDataset:
    """Load the full validated leakage-aware modeling dataset."""

    data = load_validated_dataset()
    return build_modeling_dataset(data)


def run_unweighted_sensitivity(
    *,
    outer_n_splits: int = 5,
    outer_n_repeats: int = 5,
    inner_n_splits: int = 4,
    n_resamples: int = 2000,
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold: float = 0.5,
    n_jobs: int = 1,
) -> UnweightedSensitivityResults:
    """Run nested evaluation without class weighting."""

    modeling = build_unweighted_sensitivity_dataset()

    outer_splits = build_repeated_group_splits(
        modeling.predictors,
        modeling.outcome,
        modeling.groups,
        n_splits=outer_n_splits,
        n_repeats=outer_n_repeats,
        random_state=random_state,
    )

    specifications = build_model_specifications(
        random_state=random_state,
        class_weighting=False,
    )

    evaluation = evaluate_outer_splits(
        modeling,
        outer_splits,
        specifications,
        inner_n_splits=inner_n_splits,
        threshold=threshold,
        n_jobs=n_jobs,
    )

    summary = build_summary_results(
        evaluation.predictions,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    comparisons = grouped_bootstrap_comparisons(
        evaluation.predictions,
        n_resamples=n_resamples,
        random_state=random_state,
    )

    return UnweightedSensitivityResults(
        modeling=modeling,
        evaluation=evaluation,
        summary=summary,
        comparisons=comparisons,
    )


def save_unweighted_sensitivity_results(
    results: UnweightedSensitivityResults,
    *,
    reports_directory: Path = DEFAULT_REPORTS_DIRECTORY,
) -> dict[str, Path]:
    """Save all unweighted sensitivity result tables."""

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = save_evaluation_results(
        results.evaluation,
        reports_directory=reports_directory,
    )

    paths.update(
        save_summary_results(
            results.summary,
            reports_directory=reports_directory,
        )
    )

    paths["comparisons"] = save_comparisons(
        results.comparisons,
        reports_directory=reports_directory,
    )

    return paths


def main() -> None:
    """Run and save the complete unweighted sensitivity analysis."""

    results = run_unweighted_sensitivity()
    paths = save_unweighted_sensitivity_results(results)

    modeling = results.modeling
    evaluation = results.evaluation

    print("Sensitivity analysis: no class weighting")
    print("Rows:", len(modeling.predictors))
    print("Unique groups:", modeling.groups.nunique())
    print(
        "Outcome counts:",
        modeling.outcome.value_counts().sort_index().to_dict(),
    )
    print(
        "Evaluated outer folds:",
        evaluation.fold_metrics[["repeat", "fold"]].drop_duplicates().shape[0],
    )
    print(
        "Evaluated models:",
        evaluation.fold_metrics["model"].nunique(),
    )
    print(
        "Fold metric rows:",
        len(evaluation.fold_metrics),
    )
    print(
        "Prediction rows:",
        len(evaluation.predictions),
    )
    print(
        "Performance summary rows:",
        len(results.summary.performance_summary),
    )
    print(
        "Paired comparison rows:",
        len(results.comparisons),
    )

    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
