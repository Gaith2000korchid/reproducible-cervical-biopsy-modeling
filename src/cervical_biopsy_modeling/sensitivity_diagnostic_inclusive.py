"""Run the diagnostic-inclusive leakage audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cervical_biopsy_modeling.audit import (
    LEAKAGE_COLUMNS,
    load_validated_dataset,
)
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
    build_diagnostic_inclusive_modeling_dataset,
)
from cervical_biopsy_modeling.models import build_model_specifications
from cervical_biopsy_modeling.summary import (
    SummaryResults,
    build_summary_results,
    save_summary_results,
)
from cervical_biopsy_modeling.validation import build_repeated_group_splits

DEFAULT_RANDOM_STATE = 42

DEFAULT_REPORTS_DIRECTORY = Path("reports/sensitivity_diagnostic_inclusive")


@dataclass(frozen=True)
class DiagnosticInclusiveAuditResults:
    """All results produced by the diagnostic-inclusive leakage audit."""

    modeling: ModelingDataset
    evaluation: EvaluationResults
    summary: SummaryResults
    comparisons: pd.DataFrame


def build_diagnostic_inclusive_audit_dataset() -> ModelingDataset:
    """Load validated data with diagnostic variables intentionally retained."""

    data = load_validated_dataset()

    return build_diagnostic_inclusive_modeling_dataset(data)


def run_diagnostic_inclusive_audit(
    *,
    outer_n_splits: int = 5,
    outer_n_repeats: int = 5,
    inner_n_splits: int = 4,
    n_resamples: int = 2000,
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold: float = 0.5,
    n_jobs: int = 1,
) -> DiagnosticInclusiveAuditResults:
    """Run the explicitly labelled diagnostic-inclusive leakage audit."""

    modeling = build_diagnostic_inclusive_audit_dataset()

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

    return DiagnosticInclusiveAuditResults(
        modeling=modeling,
        evaluation=evaluation,
        summary=summary,
        comparisons=comparisons,
    )


def save_diagnostic_inclusive_audit_results(
    results: DiagnosticInclusiveAuditResults,
    *,
    reports_directory: Path = DEFAULT_REPORTS_DIRECTORY,
) -> dict[str, Path]:
    """Save all diagnostic-inclusive audit result tables."""

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
    """Run and save the complete diagnostic-inclusive leakage audit."""

    results = run_diagnostic_inclusive_audit()

    paths = save_diagnostic_inclusive_audit_results(results)

    modeling = results.modeling
    evaluation = results.evaluation

    print("Leakage audit: diagnostic-inclusive predictors")

    print(
        "Warning:",
        "results are not valid pre-screening performance estimates.",
    )

    print("Rows:", len(modeling.predictors))

    print("Predictors:", modeling.predictors.shape[1])

    print(
        "Intentionally included diagnostic variables:",
        list(LEAKAGE_COLUMNS),
    )

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
