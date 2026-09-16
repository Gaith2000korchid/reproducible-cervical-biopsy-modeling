# Reproducible Cervical Biopsy Modeling

Leakage-aware prediction of concurrent cervical biopsy positivity using reproducible preprocessing and nested cross-validation.

## Project status

This project is under active development.

The modeling protocol has been specified before implementation. No performance result will be reported until the complete validation workflow has been implemented and tested.

## Research question

How well can demographic, behavioral, reproductive, and medical-history variables available before the current cervical examination discriminate between positive and negative biopsy outcomes?

The project predicts concurrent biopsy positivity. It does not predict future cervical cancer incidence.

## Dataset

The project uses the public **Cervical Cancer (Risk Factors)** dataset from the UCI Machine Learning Repository:

- 858 observations;
- 36 variables;
- binary `Biopsy` outcome;
- demographic, behavioral, reproductive, and medical-history information;
- missing values;
- CC BY 4.0 license.

Dataset DOI: https://doi.org/10.24432/C5Z310

Raw and processed datasets are not committed to this repository. They will be retrieved and generated through reproducible project code.

## Methodological principles

The primary analysis is designed around the following rules:

- exclude current diagnostic results from pre-screening predictors;
- fit imputation, scaling, and feature filtering inside training folds;
- keep exact duplicate profiles in the same cross-validation fold;
- use nested, stratified, group-aware cross-validation;
- compare a dummy baseline, regularized logistic regression, and XGBoost;
- prioritize average precision for the imbalanced outcome;
- report uncertainty and clinically relevant error types;
- separate the primary analysis from exploratory leakage audits.

The full prespecified protocol is available in [`docs/modeling_protocol.md`](docs/modeling_protocol.md).

## Repository structure

```text
.
├── artifacts/                 # Generated model artifacts, not tracked
├── data/
│   ├── raw/                   # Official source data, not tracked
│   ├── processed/             # Reproducible derived data, not tracked
│   └── README.md              # Provenance and data-governance notes
├── docs/
│   └── modeling_protocol.md   # Prespecified modeling protocol
├── notebooks/                 # Narrative analyses
├── reports/
│   └── figures/               # Final reproducible figures
├── src/
│   └── cervical_biopsy_modeling/
├── tests/                     # Automated tests
├── pyproject.toml
└── README.md
```

## Reproducibility

The project uses:

- Python 3.13;
- `uv` for dependency management;
- a `src/` package layout;
- fixed and recorded random seeds;
- automated tests;
- documented data provenance.

Installation and execution commands will be added as the workflow is implemented.

## Project origin

The topic was initially explored through a Coursera Guided Project. This repository is a new implementation with an original scientific question, original code, leakage controls, nested validation, testing, and reproducibility safeguards.

No Coursera source code or course notebook is included.

## Intended use and limitations

This repository is an educational and methodological portfolio project.

The models are not intended for diagnosis, clinical decision-making, or patient care. Results will represent internal validation on one cross-sectional dataset and will not establish clinical utility, causality, or external validity.

## License

Project code will be released under the MIT License.

The UCI dataset remains governed by its CC BY 4.0 license and must be cited separately.
