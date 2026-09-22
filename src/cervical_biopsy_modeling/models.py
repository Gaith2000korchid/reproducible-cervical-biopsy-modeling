"""Model specifications for nested validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cervical_biopsy_modeling.preprocessing import (
    build_preprocessing_pipeline,
)
from cervical_biopsy_modeling.xgboost_model import (
    BalancedXGBClassifier,
)

DEFAULT_RANDOM_STATE = 42

LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
XGBOOST_N_ESTIMATORS = (100, 300)
XGBOOST_MAX_DEPTHS = (2, 3)
XGBOOST_LEARNING_RATES = (0.03, 0.1)


@dataclass(frozen=True)
class ModelSpecification:
    """An estimator and its inner-validation parameter grid."""

    name: str
    estimator: Pipeline
    parameter_grid: dict[str, tuple[Any, ...]]


def build_classifier_pipeline(
    classifier: BaseEstimator,
    *,
    scale: bool,
    add_missing_indicators: bool = True,
) -> Pipeline:
    """Combine leakage-safe preprocessing with a classifier."""

    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessing_pipeline(
                    scale=scale,
                    add_missing_indicators=add_missing_indicators,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_dummy_specification(
    *,
    add_missing_indicators: bool = True,
) -> ModelSpecification:
    """Build the empirical-prevalence reference classifier."""

    estimator = build_classifier_pipeline(
        DummyClassifier(strategy="prior"),
        scale=False,
        add_missing_indicators=add_missing_indicators,
    )

    return ModelSpecification(
        name="dummy_prior",
        estimator=estimator,
        parameter_grid={},
    )


def build_logistic_specification(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    class_weighting: bool = True,
    add_missing_indicators: bool = True,
) -> ModelSpecification:
    """Build the optionally weighted L2 logistic classifier."""

    class_weight = "balanced" if class_weighting else None

    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=2_000,
        random_state=random_state,
    )

    estimator = build_classifier_pipeline(
        classifier,
        scale=True,
        add_missing_indicators=add_missing_indicators,
    )

    return ModelSpecification(
        name="logistic_regression",
        estimator=estimator,
        parameter_grid={
            "classifier__C": LOGISTIC_C_VALUES,
        },
    )


def build_xgboost_specification(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = 1,
    class_weighting: bool = True,
    add_missing_indicators: bool = True,
) -> ModelSpecification:
    """Build the optionally training-weighted XGBoost classifier."""

    classifier = BalancedXGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        class_weighting=class_weighting,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    estimator = build_classifier_pipeline(
        classifier,
        scale=False,
        add_missing_indicators=add_missing_indicators,
    )

    return ModelSpecification(
        name="xgboost",
        estimator=estimator,
        parameter_grid={
            "classifier__n_estimators": XGBOOST_N_ESTIMATORS,
            "classifier__max_depth": XGBOOST_MAX_DEPTHS,
            "classifier__learning_rate": XGBOOST_LEARNING_RATES,
        },
    )


def build_model_specifications(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    class_weighting: bool = True,
    add_missing_indicators: bool = True,
) -> dict[str, ModelSpecification]:
    """Build all currently implemented model specifications."""

    specifications = (
        build_dummy_specification(
            add_missing_indicators=add_missing_indicators,
        ),
        build_logistic_specification(
            random_state=random_state,
            class_weighting=class_weighting,
            add_missing_indicators=add_missing_indicators,
        ),
        build_xgboost_specification(
            random_state=random_state,
            n_jobs=1,
            class_weighting=class_weighting,
            add_missing_indicators=add_missing_indicators,
        ),
    )

    return {specification.name: specification for specification in specifications}
