# Reproduce the cervical-biopsy analysis

Run commands from the repository root. Python 3.13 and `uv` are required.

```bash
uv sync --frozen
uv run --frozen pytest --cov=cervical_biopsy_modeling
uv run --frozen python -m cervical_biopsy_modeling.reproduce --mode figures --output-dir artifacts/figure-rerun
uv run --frozen python -m cervical_biopsy_modeling.reproduce --mode full --output-dir artifacts/full-rerun
```

On macOS, install `libomp` with Homebrew if XGBoost cannot load its OpenMP runtime. Linux GitHub Actions requires no Homebrew configuration.

## What each mode verifies

| Mode | Inputs | Work | Limits |
|---|---|---|---|
| `figures` | Published CSV tables | Regenerate all four figures | Does not download data or refit models |
| `primary` | Checksum-validated UCI data | Audit, grouped splits, nested evaluation, summaries and paired comparisons | Does not run the four sensitivities or combined figures |
| `full` | Checksum-validated UCI data | Primary analysis, four sensitivities, paired comparisons and figures | Internal validation only; bootstrap intervals remain conditional on fitted out-of-fold predictions |

Every output directory must be empty. Published `reports/` files are preserved. A new directory contains `logs/`, `reports/` and `run_record.json`; primary/full modes also contain downloaded `data/raw/`. The run record captures the source commit, dirty-state indicator, code/dependency-lock hashes, software versions, step exit codes, dataset fingerprint and result-table hashes. An interrupted or failed stage is marked incomplete; inspect its log before interpreting any partial output.

The primary steps run in this order: `data`, `audit`, `validation`, `evaluation`, `summary`, `comparisons`. Full mode then runs duplicate removal, unweighted models and their paired comparison, diagnostic-inclusive models and their paired comparison, models without missingness indicators and their paired comparison, followed by `reporting`.

The settings remain the documented five repeated outer folds, four inner folds, fixed threshold 0.5 and 2,000 grouped bootstrap resamples. Seeds and dataset checksum are defined in the analysis modules. Reproducing a run does not demonstrate external validity, causality or clinical usefulness.

## GitHub Actions

Automatic CI checks tests, coverage and figure generation. In the Actions tab, choose **Tests and reproducible figures**, **Run workflow**, and enable **full_analysis** to execute the entire analysis. Full execution is optional because it refits models and performs repeated bootstrap calculations; it is not silently performed by the quick figure check. Artifacts include result tables and run records; the downloaded raw CSV is omitted from the upload.
