"""Tests for reproducible raw-data auditing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cervical_biopsy_modeling.audit import (
    DataAudit,
    audit_dataframe,
    make_predictor_groups,
    primary_predictor_columns,
    write_audit_report,
)
from cervical_biopsy_modeling.data import DatasetValidationError


def make_audit_dataframe() -> pd.DataFrame:
    """Create a small dataset containing one conflicting predictor group."""
    return pd.DataFrame(
        {
            "Age": [20, 20, 30],
            "First sexual intercourse": [21, 21, 18],
            "risk_feature": [2.0, 2.0, 1.0],
            "nullable_feature": [float("nan"), float("nan"), 0.0],
            "Hinselmann": [0, 0, 0],
            "Schiller": [0, 0, 0],
            "Citology": [0, 0, 0],
            "Dx:Cancer": [0, 0, 0],
            "Dx:CIN": [0, 0, 0],
            "Dx:HPV": [0, 0, 0],
            "Dx": [0, 0, 0],
            "Biopsy": [0, 1, 0],
        }
    )


def test_primary_predictors_exclude_outcome_and_leakage() -> None:
    predictors = primary_predictor_columns(make_audit_dataframe())

    assert predictors == [
        "Age",
        "First sexual intercourse",
        "risk_feature",
        "nullable_feature",
    ]


def test_identical_profiles_receive_same_group() -> None:
    groups = make_predictor_groups(make_audit_dataframe())

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_audit_detects_conflicting_group_and_age_inconsistencies() -> None:
    audit = audit_dataframe(make_audit_dataframe())

    assert audit.rows == 3
    assert audit.primary_predictor_count == 4
    assert audit.outcome_counts == {"0": 2, "1": 1}
    assert audit.duplicate_predictor_occurrences_beyond_first == 1
    assert audit.duplicate_predictor_groups == 1
    assert audit.largest_predictor_group == 2
    assert audit.groups_with_conflicting_outcomes == 1
    assert audit.first_intercourse_after_current_age == 2


def test_primary_predictors_require_all_prespecified_columns() -> None:
    data = make_audit_dataframe().drop(columns="Dx")

    with pytest.raises(DatasetValidationError, match="Dx"):
        primary_predictor_columns(data)


def test_write_audit_report_creates_valid_json(tmp_path: Path) -> None:
    audit = audit_dataframe(make_audit_dataframe())
    report_path = tmp_path / "audit.json"

    written_path = write_audit_report(audit, report_path)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert written_path == report_path
    assert payload["rows"] == 3
    assert written_path.read_text(encoding="utf-8").endswith("\n")
    assert isinstance(audit, DataAudit)
