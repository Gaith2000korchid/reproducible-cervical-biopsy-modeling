"""Create a reproducible audit of the raw cervical biopsy dataset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from cervical_biopsy_modeling.data import (
    CSV_FILENAME,
    DEFAULT_RAW_DIR,
    DatasetValidationError,
    validate_csv_content,
)

OUTCOME_COLUMN = "Biopsy"
LEAKAGE_COLUMNS = (
    "Hinselmann",
    "Schiller",
    "Citology",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
)
MISSINGNESS_THRESHOLD = 0.80

DEFAULT_DATA_PATH = DEFAULT_RAW_DIR / CSV_FILENAME
DEFAULT_REPORT_PATH = Path("reports/data_audit.json")


@dataclass(frozen=True)
class DataAudit:
    """Structured results from the raw-data audit."""

    rows: int
    columns: int
    outcome_counts: dict[str, int]
    outcome_prevalence: float
    missing_values: int
    exact_duplicate_rows_beyond_first: int
    primary_predictor_count: int
    duplicate_predictor_occurrences_beyond_first: int
    duplicate_predictor_groups: int
    largest_predictor_group: int
    groups_with_conflicting_outcomes: int
    high_missingness_columns: dict[str, float]
    constant_columns: list[str]
    first_intercourse_after_current_age: int


def load_validated_dataset(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a local dataset only after checksum and contract validation."""
    dataset_path = Path(path)
    content = dataset_path.read_bytes()
    validate_csv_content(content)
    return pd.read_csv(dataset_path, na_values="?")


def primary_predictor_columns(data: pd.DataFrame) -> list[str]:
    """Return variables admissible at the prespecified prediction time."""
    excluded_columns = {OUTCOME_COLUMN, *LEAKAGE_COLUMNS}
    missing_columns = excluded_columns.difference(data.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(
            f"Required outcome or leakage columns are missing: {missing_names}."
        )

    return [column for column in data.columns if column not in excluded_columns]


def make_predictor_groups(data: pd.DataFrame) -> pd.Series:
    """Assign the same group to identical primary predictor profiles."""
    predictors = primary_predictor_columns(data)
    groups = data.groupby(
        predictors,
        dropna=False,
        sort=True,
    ).ngroup()

    return groups.astype("int64").rename("predictor_group")


def audit_dataframe(data: pd.DataFrame) -> DataAudit:
    """Compute prespecified data-quality and leakage diagnostics."""
    predictors = primary_predictor_columns(data)
    groups = make_predictor_groups(data)

    grouped_data = data.groupby(groups, sort=False)
    group_sizes = grouped_data.size()
    conflicting_outcomes = grouped_data[OUTCOME_COLUMN].nunique().gt(1)

    missingness = data.isna().mean()
    high_missingness = missingness[missingness > MISSINGNESS_THRESHOLD].sort_values(
        ascending=False
    )

    outcome_counts = data[OUTCOME_COLUMN].value_counts().sort_index()

    return DataAudit(
        rows=len(data),
        columns=len(data.columns),
        outcome_counts={
            str(int(value)): int(count) for value, count in outcome_counts.items()
        },
        outcome_prevalence=float(data[OUTCOME_COLUMN].mean()),
        missing_values=int(data.isna().sum().sum()),
        exact_duplicate_rows_beyond_first=int(data.duplicated().sum()),
        primary_predictor_count=len(predictors),
        duplicate_predictor_occurrences_beyond_first=int(
            group_sizes.sub(1).clip(lower=0).sum()
        ),
        duplicate_predictor_groups=int(group_sizes.gt(1).sum()),
        largest_predictor_group=int(group_sizes.max()),
        groups_with_conflicting_outcomes=int(conflicting_outcomes.sum()),
        high_missingness_columns={
            str(column): float(value) for column, value in high_missingness.items()
        },
        constant_columns=[
            column for column in data.columns if data[column].nunique(dropna=True) <= 1
        ],
        first_intercourse_after_current_age=int(
            (data["First sexual intercourse"] > data["Age"]).sum()
        ),
    )


def write_audit_report(audit: DataAudit, path: Path | str) -> Path:
    """Write the audit as deterministic, human-readable JSON."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    """Run the raw-data audit from the command line."""
    parser = argparse.ArgumentParser(
        description="Audit the validated UCI cervical biopsy dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the validated raw CSV.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path for the generated JSON report.",
    )
    args = parser.parse_args()

    data = load_validated_dataset(args.data_path)
    audit = audit_dataframe(data)
    report_path = write_audit_report(audit, args.report_path)

    print(f"Audit report: {report_path}")
    print(f"Rows: {audit.rows}")
    print(f"Columns: {audit.columns}")
    print(f"Biopsy-positive prevalence: {audit.outcome_prevalence:.4%}")
    print(
        "Conflicting predictor groups:",
        audit.groups_with_conflicting_outcomes,
    )


if __name__ == "__main__":
    main()
