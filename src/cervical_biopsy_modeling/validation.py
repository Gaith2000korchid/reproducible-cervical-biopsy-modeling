"""Build and audit repeated stratified group-aware validation splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cervical_biopsy_modeling.audit import load_validated_dataset
from cervical_biopsy_modeling.modeling_data import build_modeling_dataset

DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_REPORT_PATH = Path("reports/outer_cv_splits.csv")


@dataclass(frozen=True)
class GroupedSplit:
    """Indices and identifiers for one group-aware evaluation fold."""

    repeat: int
    fold: int
    random_state: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def validate_split(
    outcome: pd.Series,
    groups: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Validate isolation of indices, groups, and outcome classes."""
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Training and test indices overlap.")

    train_groups = set(groups.iloc[train_indices].tolist())
    test_groups = set(groups.iloc[test_indices].tolist())

    if train_groups.intersection(test_groups):
        raise ValueError("Training and test groups overlap.")

    if outcome.iloc[train_indices].nunique() != 2:
        raise ValueError("A training fold does not contain both outcome classes.")

    if outcome.iloc[test_indices].nunique() != 2:
        raise ValueError("A test fold does not contain both outcome classes.")


def build_repeated_group_splits(
    predictors: pd.DataFrame,
    outcome: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[GroupedSplit]:
    """Create reproducible repeated stratified group-aware splits."""
    if not predictors.index.equals(outcome.index):
        raise ValueError("Predictor and outcome indices do not match.")

    if not predictors.index.equals(groups.index):
        raise ValueError("Predictor and group indices do not match.")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    if groups.nunique() < n_splits:
        raise ValueError("The number of groups must be at least n_splits.")

    if outcome.nunique() != 2:
        raise ValueError("The outcome must contain exactly two classes.")

    generated_splits: list[GroupedSplit] = []

    for repeat_index in range(n_repeats):
        repeat_seed = random_state + repeat_index
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=repeat_seed,
        )

        repeat_test_indices: list[np.ndarray] = []

        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(predictors, outcome, groups)
        ):
            validate_split(
                outcome,
                groups,
                train_indices,
                test_indices,
            )

            repeat_test_indices.append(test_indices)
            generated_splits.append(
                GroupedSplit(
                    repeat=repeat_index + 1,
                    fold=fold_index + 1,
                    random_state=repeat_seed,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )

        observed_test_indices = np.sort(np.concatenate(repeat_test_indices))
        expected_test_indices = np.arange(len(predictors))

        if not np.array_equal(observed_test_indices, expected_test_indices):
            raise ValueError(
                "Every observation must appear exactly once in the test "
                "folds of each repeat."
            )

    return generated_splits


def summarize_splits(
    splits: list[GroupedSplit],
    outcome: pd.Series,
    groups: pd.Series,
) -> pd.DataFrame:
    """Return a tabular audit of all generated evaluation folds."""
    records: list[dict[str, int | float]] = []

    for split in splits:
        train_outcome = outcome.iloc[split.train_indices]
        test_outcome = outcome.iloc[split.test_indices]

        records.append(
            {
                "repeat": split.repeat,
                "fold": split.fold,
                "random_state": split.random_state,
                "train_size": len(split.train_indices),
                "test_size": len(split.test_indices),
                "train_groups": groups.iloc[split.train_indices].nunique(),
                "test_groups": groups.iloc[split.test_indices].nunique(),
                "train_positives": int(train_outcome.sum()),
                "test_positives": int(test_outcome.sum()),
                "test_negatives": int((test_outcome == 0).sum()),
                "test_prevalence": float(test_outcome.mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    """Generate and save the outer cross-validation split audit."""
    parser = argparse.ArgumentParser(
        description="Generate repeated stratified group-aware outer folds."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    args = parser.parse_args()

    data = load_validated_dataset()
    modeling_data = build_modeling_dataset(data)

    splits = build_repeated_group_splits(
        modeling_data.predictors,
        modeling_data.outcome,
        modeling_data.groups,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    summary = summarize_splits(
        splits,
        modeling_data.outcome,
        modeling_data.groups,
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.report_path, index=False)

    print(f"Split report: {args.report_path}")
    print(f"Evaluation folds: {len(summary)}")
    print(
        "Test positives per fold:",
        f"{summary['test_positives'].min()}–{summary['test_positives'].max()}",
    )


if __name__ == "__main__":
    main()
