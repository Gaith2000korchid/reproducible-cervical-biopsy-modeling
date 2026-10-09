# Full reproduction validation

This note records one complete refit. It is not a replacement of the published tables and it is not an external clinical validation.

## Execution

- Analyzed commit: `41266a77b6b78734f483641138f5d713c55c4f74`
- Workflow: `Tests and reproducible figures`, manual dispatch with `full_analysis=true`
- Run: https://github.com/Gaith2000korchid/reproducible-cervical-biopsy-modeling/actions/runs/37951848145
- Checkout: clean (`source_dirty=false`)
- Environment: GitHub-hosted Ubuntu, Python 3.13.16, frozen `uv.lock`
- Packages recorded in `run_record.json`: NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, XGBoost 3.4.1, matplotlib 3.11.2
- Started: 2026-10-09T15:28:14Z
- Finished: 2026-10-09T16:39:31Z
- Observed duration: 1 hour 11 minutes 17 seconds
- Status: complete; all 14 steps exited 0

The same run first executed the automatic checks: frozen install, pytest with coverage, and figure regeneration from the published tables. Those checks do not refit models. The full job then downloaded the UCI file into an empty directory and refitted the primary analysis plus the four sensitivities.

## Steps covered

`data`, `audit`, `validation`, `evaluation`, `summary`, `comparisons`, `sensitivity_duplicates`, `sensitivity_unweighted`, `weighting_comparisons`, `sensitivity_diagnostic_inclusive`, `diagnostic_comparisons`, `sensitivity_no_missing_indicators`, `missing_indicator_comparisons`, `reporting`.

The downloaded CSV matched the expected SHA-256 `8df193ad5c9ff4288fb4c401eef70dcd2cbda404ce7f82ac74c68cfc960ab063` and the expected 858 by 36 shape. The audit reports 803 negative and 55 positive biopsy outcomes, 28 primary predictors and 22 duplicate predictor groups. Fold 1 contains 665 training groups and 166 test groups. The 25 outer splits match the published split table.

## Comparison with published tables

Dummy-prior and logistic-regression ROC AUC match the published primary values exactly: 0.4992528019925281 and 0.6133205026604778, including the published bootstrap bounds. Out-of-fold identifiers align on repeat, fold, model and row position (12,870 prediction rows). Logistic predicted classes match; the largest absolute probability difference is 0.000479. One logistic inner score differs in the fourth decimal and does not change the published primary ROC AUC.

XGBoost does not reproduce exactly on this Linux runner. Primary ROC AUC is 0.5874606588927884 against the published 0.593159741877052 (difference -0.005699). 12 of the 25 published XGBoost hyperparameter selections differ (compared as parsed JSON parameter objects), and 264 of 4,290 XGBoost predicted classes differ. Numerical differences also appear in the sensitivities, with both signs:

| Analysis | Published XGBoost ROC AUC | Recalculated | Difference |
|---|---:|---:|---:|
| Primary | 0.593159741877052 | 0.5874606588927884 | -0.005699 |
| Duplicate removal | 0.627443371974584 | 0.6223723141097153 | -0.005071 |
| Unweighted | 0.6302954828484093 | 0.6333159741877052 | +0.003020 |
| Diagnostic-inclusive | 0.9436861768368618 | 0.9423706554964337 | -0.001316 |
| No missing indicators | 0.5734133363523152 | 0.5652100079248273 | -0.008203 |

Logistic ROC AUC is exact in every analysis except the unweighted sensitivity, where it moves from 0.5950888712781613 to 0.595052643495981 (difference -0.000036). Dummy-prior values match in every analysis. Published tables were not overwritten.

The protocol, seeds, splits and dependency lock were unchanged. The published tables do not record the original host. This rerun is Ubuntu with XGBoost 3.4.1. The observed differences are concentrated in XGBoost inner selection and tree fitting, with a small logistic difference in the unweighted sensitivity. These observations do not establish the numerical cause: the original host was not recorded, and no controlled host/thread experiment was performed. They are not evidence of a better or worse clinical model.

## Artefacts

The run uploads `full-nested-validation` (reports, logs and `run_record.json`) and `verified-figures-and-coverage`. `actions/upload-artifact@v4` keeps artefacts for 90 days by default. Raw downloaded data are not uploaded. This documentation commit is later than the analyzed commit and was not part of the refit.

## Independent artifact verification and durable evidence

The downloaded ZIP SHA-256 matches GitHub's artifact digest. All 39 result-table hashes in the run record were checked against the extracted files. The 25 split-summary rows match the published table; primary predictions align on repeat, fold, model and row position (12,870 unique rows). Predicted-class differences are 0 for the dummy baseline, 0 for logistic regression and 264 for XGBoost.

An independent comparison corrects the previous claim of 25 changed XGBoost selections: **12 of 25 parameter objects differ**. The maximum primary probability differences are 0 for the dummy, 0.0004789506498081575 for logistic regression and 0.6280224397778511 for XGBoost. Small aggregate AUC differences do not imply that individual predictions are interchangeable.

Compact evidence is preserved in [reproduction/2026-10-09](reproduction/2026-10-09/):
- [Original run record](reproduction/2026-10-09/run_record.json), retained byte for byte.
- [All summary-metric comparisons](reproduction/2026-10-09/metric_comparison.csv), covering all five analyses.
- [Verification manifest](reproduction/2026-10-09/manifest.json), with file hashes and artifact provenance.

This verification inspected recorded outputs; it did not refit models, redownload the raw dataset or independently establish the origin of the published tables. The dataset hash and software versions are recorded execution evidence. The full artifact expires on 7 January 2027; the compact files above remain versioned. Published result tables are unchanged.
