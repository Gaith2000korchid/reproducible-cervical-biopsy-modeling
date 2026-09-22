# Modeling Protocol

## Status

Version 0.1 — specified before implementing the modeling workflow.

Any material change made after model evaluation must be documented with its rationale in the project history.

## Scientific question

Among participants included in the UCI Cervical Cancer (Risk Factors) dataset, how well can demographic, behavioral, reproductive, and medical-history variables available before the current cervical examination discriminate between positive and negative biopsy outcomes?

This project evaluates concurrent biopsy positivity. It does not predict future cervical cancer incidence and does not establish causal relationships.

## Intended use

This is an educational and methodological project designed to demonstrate:

- leakage-aware predictive modeling;
- reproducible preprocessing;
- appropriate evaluation for imbalanced clinical data;
- nested cross-validation;
- transparent reporting of limitations.

The resulting models are not intended for diagnosis, clinical decision-making, or patient care.

## Data source

The project uses the official UCI Machine Learning Repository dataset:

- Dataset: Cervical Cancer (Risk Factors)
- UCI identifier: 383
- DOI: 10.24432/C5Z310
- Source institution: Hospital Universitario de Caracas
- Sample size: 858 observations
- License: CC BY 4.0

The dataset will be retrieved from its official source through reproducible code. Raw and processed data files will not be committed to the repository.

## Outcome

The binary outcome is:

- `Biopsy = 1`: positive biopsy;
- `Biopsy = 0`: negative biopsy.

`Biopsy` is the only prediction target in the primary analysis.

## Prediction time

The prediction time is defined as immediately before the current cervical diagnostic examinations represented in the dataset.

Only information that could reasonably be available at this time may be used as a predictor in the primary model.

## Leakage exclusions

The following variables will be excluded from the primary predictor set because they represent current examination results or existing diagnostic information:

- `Hinselmann`;
- `Schiller`;
- `Citology`;
- `Dx:Cancer`;
- `Dx:CIN`;
- `Dx:HPV`;
- `Dx`.

These variables may be used only in a clearly labelled leakage audit. Results from that audit will not be presented as valid pre-screening performance.

## Data-quality rules

Question marks and equivalent placeholders will be interpreted as missing values.

Rows will not be removed solely because they contain missing values.

Features with more than 80% missing values will be excluded according to a predefined rule.

Impossible or internally inconsistent values will be reported. They will not be silently corrected or removed.

Constant predictors will be removed using training data only.

## Duplicate observations

The dataset does not provide a reliable patient identifier.

Exact duplicate predictor profiles will therefore receive a shared group identifier and will be assigned to the same cross-validation fold. This prevents identical profiles from being divided between training and evaluation data.

A sensitivity analysis will repeat the evaluation after retaining one observation per exact duplicate group.

## Preprocessing

All learned preprocessing operations will be fitted exclusively on the corresponding training fold.

The primary preprocessing strategy will include:

- median imputation for numeric predictors;
- missingness indicators when informative;
- standardization for linear models;
- no preprocessing fitted on the complete dataset before cross-validation.

Synthetic oversampling will not be used in the primary analysis. Class weighting will be preferred because the number of positive outcomes is small.

## Candidate models

The primary comparison will include:

1. a `DummyClassifier` baseline;
2. regularized logistic regression;
3. an XGBoost classifier.

The baseline establishes the performance expected without useful predictive information.

Logistic regression provides an interpretable statistical baseline, while XGBoost evaluates whether nonlinear relationships improve discrimination.

## Validation design

Model evaluation will use nested, group-aware, stratified cross-validation.

- Outer evaluation: 5 folds repeated 5 times;
- Inner model selection: 4 folds;
- Grouping: exact duplicate predictor profiles;
- Stratification: biopsy outcome;
- Reproducibility: fixed and recorded random seeds.

Hyperparameters and any data-dependent decision will be selected using only the inner training process. Outer evaluation folds will not be used for model or threshold selection.

Because the dataset is small and contains few positive outcomes, no separate holdout set will be removed from the primary analysis.

## Evaluation metrics

The primary metric is average precision, also called area under the precision-recall curve, because biopsy-positive observations are rare.

Secondary metrics are:

- ROC AUC;
- sensitivity;
- specificity;
- positive predictive value;
- F1 score;
- balanced accuracy;
- Brier score.

Threshold-dependent metrics will first be reported at the fixed probability threshold of 0.5.

A sensitivity-oriented threshold may be explored only if it is selected within the training process and then evaluated on unseen outer folds.

## Uncertainty

Performance will not be summarized by a single score alone.

The report will include:

- variation across outer folds and repetitions;
- participant-level out-of-fold predictions;
- uncertainty intervals when statistically appropriate;
- the number of positive observations available in each evaluation set.

### Uncertainty implementation note

Added on 2026-09-17 after validating the evaluation pipeline and before inspecting aggregate performance results.

Primary point estimates will be computed within each complete outer repetition by pooling the 858 out-of-fold predictions. The report will summarize the mean, standard deviation, and range across the five repetitions. Individual outer folds will not be treated as independent performance estimates.

Conditional 95% uncertainty intervals will be estimated by resampling exact-predictor groups with replacement. The same bootstrap sample will be applied across models and repetitions to preserve pairing. Within each bootstrap sample, metrics will be computed separately for each repetition and then averaged across repetitions.

These intervals quantify uncertainty conditional on the observed out-of-fold predictions. They do not account for every source of model-training uncertainty and must not be interpreted as external-validation confidence intervals.

## Planned sensitivity analyses

The following analyses are planned:

1. grouped duplicates versus one observation per duplicate group;
2. class weighting versus no class weighting;
3. primary leakage-aware predictors versus a diagnostic-inclusive leakage audit;
4. alternative missingness handling where justified.

These analyses will be labelled exploratory and kept separate from the primary result.

### Missingness sensitivity implementation note

Added on 2026-09-22 before running the alternative missingness sensitivity analysis.

This analysis will retain the primary training-fold missingness filter and median imputation, but will disable the addition of missingness-indicator features. All other predictors, grouping rules, validation splits, model specifications, class-weighting settings, thresholds, metrics, and uncertainty calculations will remain identical to the primary analysis.

This comparison isolates the contribution of missingness indicators without introducing a more complex imputation method that may be unstable given the small number of biopsy-positive observations.
## Reproducibility requirements

The project will include:

- a pinned Python environment and `uv.lock`;
- fixed random seeds;
- reusable code under `src/`;
- automated tests;
- documented data provenance;
- commands to reproduce the analysis;
- generated figures derived only from public data.

No Coursera code, private internship code, confidential data, credentials, or API keys will be included.

## Interpretation limits

Performance estimates represent internal validation on a single cross-sectional dataset.

They do not demonstrate:

- external validity;
- prospective performance;
- clinical utility;
- causal effects;
- transportability to other populations or healthcare settings.

External and prospective validation would be required before any clinical use.
