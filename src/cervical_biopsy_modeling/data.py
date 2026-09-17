"""Download and validate the public UCI cervical cancer dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/383/"
    "cervical%2Bcancer%2Brisk%2Bfactors.zip"
)
CSV_FILENAME = "risk_factors_cervical_cancer.csv"
DEFAULT_RAW_DIR = Path("data/raw")

EXPECTED_ROWS = 858
EXPECTED_COLUMNS = 36
EXPECTED_CSV_SHA256 = "8df193ad5c9ff4288fb4c401eef70dcd2cbda404ce7f82ac74c68cfc960ab063"


class DatasetValidationError(ValueError):
    """Raised when the downloaded dataset does not match expectations."""


@dataclass(frozen=True)
class DownloadResult:
    """Information about a validated local dataset."""

    path: Path
    sha256: str


def compute_sha256(content: bytes) -> str:
    """Return the SHA-256 checksum of binary content."""
    return hashlib.sha256(content).hexdigest()


def validate_dataframe(data: pd.DataFrame) -> None:
    """Validate the expected shape, outcome, and outcome values."""
    expected_shape = (EXPECTED_ROWS, EXPECTED_COLUMNS)

    if data.shape != expected_shape:
        raise DatasetValidationError(
            f"Expected shape {expected_shape}, received {data.shape}."
        )

    if "Biopsy" not in data.columns:
        raise DatasetValidationError("Required outcome column 'Biopsy' is missing.")

    if data["Biopsy"].isna().any():
        raise DatasetValidationError("The Biopsy outcome contains missing values.")

    observed_values = set(data["Biopsy"].unique().tolist())
    if observed_values != {0.0, 1.0}:
        raise DatasetValidationError(
            "Biopsy must contain exactly the binary values 0 and 1; "
            f"received {sorted(observed_values)}."
        )


def validate_csv_content(content: bytes) -> str:
    """Validate CSV bytes and return their SHA-256 checksum."""
    checksum = compute_sha256(content)

    if EXPECTED_CSV_SHA256 is not None and checksum != EXPECTED_CSV_SHA256:
        raise DatasetValidationError(
            "The dataset checksum does not match the pinned reference."
        )

    data = pd.read_csv(io.BytesIO(content), na_values="?")
    validate_dataframe(data)
    return checksum


def extract_csv_content(archive_content: bytes) -> bytes:
    """Read the expected CSV from the official ZIP archive."""
    with zipfile.ZipFile(io.BytesIO(archive_content)) as archive:
        matching_members = [
            member for member in archive.namelist() if Path(member).name == CSV_FILENAME
        ]

        if len(matching_members) != 1:
            raise DatasetValidationError(
                f"Expected one {CSV_FILENAME!r} file in the archive; "
                f"found {len(matching_members)}."
            )

        return archive.read(matching_members[0])


def download_dataset(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    force: bool = False,
) -> DownloadResult:
    """Download, validate, and save the official dataset."""
    destination_directory = Path(raw_dir)
    destination = destination_directory / CSV_FILENAME

    if destination.exists() and not force:
        existing_content = destination.read_bytes()
        checksum = validate_csv_content(existing_content)
        return DownloadResult(path=destination, sha256=checksum)

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "cervical-biopsy-modeling/0.1.0"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        archive_content = response.read()

    csv_content = extract_csv_content(archive_content)
    checksum = validate_csv_content(csv_content)

    destination_directory.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_content)

    return DownloadResult(path=destination, sha256=checksum)


def main() -> None:
    """Run the dataset download from the command line."""
    parser = argparse.ArgumentParser(
        description="Download and validate UCI dataset 383."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory in which the raw CSV will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the dataset again even if it already exists.",
    )
    args = parser.parse_args()

    result = download_dataset(args.raw_dir, force=args.force)

    print(f"Dataset: {result.path}")
    print(f"SHA-256: {result.sha256}")


if __name__ == "__main__":
    main()
