"""Reproduce the analysis in a separate directory, preserving published results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

PRIMARY = ("data", "audit", "validation", "evaluation", "summary", "comparisons")
SENSITIVITIES = (
    "sensitivity_duplicates",
    "sensitivity_unweighted",
    "weighting_comparisons",
    "sensitivity_diagnostic_inclusive",
    "diagnostic_comparisons",
    "sensitivity_no_missing_indicators",
    "missing_indicator_comparisons",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(source: Path, output: Path, mode: str) -> None:
    """Run checked subprocesses and record input, code and output fingerprints."""
    source, output = source.resolve(), output.resolve()
    if mode not in {"figures", "primary", "full"}:
        raise ValueError("Mode must be figures, primary or full.")
    if not (source / "reports/performance_summary.csv").is_file():
        raise ValueError("Source must be the repository root.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    if mode == "figures":
        shutil.copytree(source / "reports", output / "reports")
        modules = ("reporting",)
    elif mode == "primary":
        modules = PRIMARY
    else:
        modules = (*PRIMARY, *SENSITIVITIES, "reporting")

    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    code_paths = sorted((source / "src").rglob("*.py"))
    code_paths += [source / "pyproject.toml", source / "uv.lock"]
    record = {
        "mode": mode,
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "source_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "source_sha256": {
            str(path.relative_to(source)): sha256(path) for path in code_paths
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib")
        },
        "interpretation": (
            "figures reuses published tables; primary/full refit models. "
            "Neither mode provides external clinical validation."
        ),
        "steps": [],
        "completed": False,
    }
    try:
        for module in modules:
            command = [sys.executable, "-m", f"cervical_biopsy_modeling.{module}"]
            print(f"Running {module} ...", flush=True)
            with (output / "logs" / f"{module}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=output,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["steps"].append({"module": module, "exit_code": result.returncode})
            result.check_returncode()
        record["completed"] = True
    finally:
        record["finished_utc"] = datetime.now(UTC).isoformat()
        raw = output / "data/raw/risk_factors_cervical_cancer.csv"
        if raw.exists():
            record["dataset_sha256"] = sha256(raw)
        record["result_sha256"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted((output / "reports").rglob("*.csv"))
        }
        (output / "run_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("figures", "primary", "full"), default="full"
    )
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/reproduction")
    )
    args = parser.parse_args()
    reproduce(args.source, args.output_dir, args.mode)
    print(f"Completed {args.mode} reproduction: {args.output_dir}")


if __name__ == "__main__":
    main()
