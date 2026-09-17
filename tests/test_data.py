"""Tests for dataset retrieval and validation utilities."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

import cervical_biopsy_modeling.data as data_module
from cervical_biopsy_modeling.data import (
    CSV_FILENAME,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    DatasetValidationError,
    compute_sha256,
    extract_csv_content,
    validate_dataframe,
)


def make_valid_dataframe() -> pd.DataFrame:
    """Create a synthetic dataframe matching the expected dataset contract."""
    data = pd.DataFrame(
        {
            f"feature_{index}": [0.0] * EXPECTED_ROWS
            for index in range(EXPECTED_COLUMNS - 1)
        }
    )
    data["Biopsy"] = [0.0] * (EXPECTED_ROWS - 1) + [1.0]
    return data


def test_compute_sha256_known_value() -> None:
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    assert compute_sha256(b"hello") == expected


def test_validate_dataframe_accepts_expected_contract() -> None:
    validate_dataframe(make_valid_dataframe())


def test_validate_dataframe_rejects_wrong_shape() -> None:
    data = make_valid_dataframe().iloc[:-1]

    with pytest.raises(DatasetValidationError, match="Expected shape"):
        validate_dataframe(data)


def test_validate_dataframe_rejects_missing_outcome() -> None:
    data = make_valid_dataframe().rename(columns={"Biopsy": "not_biopsy"})

    with pytest.raises(DatasetValidationError, match="Biopsy"):
        validate_dataframe(data)


def test_validate_dataframe_rejects_nonbinary_outcome() -> None:
    data = make_valid_dataframe()
    data.loc[0, "Biopsy"] = 2.0

    with pytest.raises(DatasetValidationError, match="binary values"):
        validate_dataframe(data)


def test_extract_csv_content_from_nested_archive() -> None:
    expected_content = b"feature,Biopsy\n1,0\n"
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr(f"nested/{CSV_FILENAME}", expected_content)

    assert extract_csv_content(archive_buffer.getvalue()) == expected_content


def test_extract_csv_content_rejects_missing_file() -> None:
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("unexpected.csv", b"value\n1\n")

    with pytest.raises(DatasetValidationError, match="Expected one"):
        extract_csv_content(archive_buffer.getvalue())


def test_validate_csv_content_rejects_unexpected_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "EXPECTED_CSV_SHA256", "0" * 64)

    with pytest.raises(DatasetValidationError, match="checksum"):
        data_module.validate_csv_content(b"unexpected content")
