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

DEFAULT_RANDOM_STATE = 42
LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)


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
) -> Pipeline:
    """Combine leakage-safe preprocessing with a classifier."""
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessing_pipeline(scale=scale),
            ),
            ("classifier", classifier),
        ]
    )


def build_dummy_specification() -> ModelSpecification:
    """Build the empirical-prevalence reference classifier."""
    estimator = build_classifier_pipeline(
        DummyClassifier(strategy="prior"),
        scale=False,
    )

    return ModelSpecification(
        name="dummy_prior",
        estimator=estimator,
        parameter_grid={},
    )


def build_logistic_specification(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ModelSpecification:
    """Build the weighted L2-regularized logistic classifier."""
    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2_000,
        random_state=random_state,
    )

    estimator = build_classifier_pipeline(
        classifier,
        scale=True,
    )

    return ModelSpecification(
        name="logistic_regression",
        estimator=estimator,
        parameter_grid={
            "classifier__C": LOGISTIC_C_VALUES,
        },
    )


def build_model_specifications(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, ModelSpecification]:
    """Build all currently implemented model specifications."""
    specifications = (
        build_dummy_specification(),
        build_logistic_specification(
            random_state=random_state,
        ),
    )

    return {specification.name: specification for specification in specifications}
