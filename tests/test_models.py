"""Tests for model specifications."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from cervical_biopsy_modeling.models import (
    LOGISTIC_C_VALUES,
    XGBOOST_LEARNING_RATES,
    XGBOOST_MAX_DEPTHS,
    XGBOOST_N_ESTIMATORS,
    build_dummy_specification,
    build_logistic_specification,
    build_model_specifications,
    build_xgboost_specification,
)
from cervical_biopsy_modeling.xgboost_model import (
    BalancedXGBClassifier,
)


def make_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a small imbalanced dataset with missing values."""

    predictors = pd.DataFrame(
        {
            "age": [
                18,
                21,
                23,
                25,
                28,
                31,
                34,
                37,
                40,
                43,
                46,
                49,
            ],
            "exposure": [
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                1,
            ],
            "partly_missing": [
                1.0,
                np.nan,
                2.0,
                2.0,
                np.nan,
                3.0,
                3.0,
                4.0,
                np.nan,
                4.0,
                5.0,
                5.0,
            ],
            "constant": [1] * 12,
        }
    )

    outcome = pd.Series(
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        name="Biopsy",
        dtype=int,
    )

    return predictors, outcome


def test_model_registry_contains_expected_models() -> None:
    """The registry must expose every implemented model."""

    specifications = build_model_specifications()

    assert set(specifications) == {
        "dummy_prior",
        "logistic_regression",
        "xgboost",
    }
    assert specifications["dummy_prior"].name == "dummy_prior"
    assert specifications["logistic_regression"].name == "logistic_regression"
    assert specifications["xgboost"].name == "xgboost"


def test_dummy_specification_uses_empirical_prior() -> None:
    """The dummy model must return the training prevalence."""

    predictors, outcome = make_classification_data()
    specification = build_dummy_specification()
    classifier = specification.estimator.named_steps["classifier"]

    assert isinstance(classifier, DummyClassifier)
    assert classifier.strategy == "prior"
    assert specification.parameter_grid == {}

    specification.estimator.fit(predictors, outcome)
    probabilities = specification.estimator.predict_proba(predictors)[:, 1]

    assert np.allclose(probabilities, outcome.mean())


def test_logistic_specification_is_weighted_and_scaled_by_default() -> None:
    """The primary logistic model must use weighting and scaling."""

    specification = build_logistic_specification(random_state=123)

    preprocessing = specification.estimator.named_steps["preprocessing"]
    classifier = specification.estimator.named_steps["classifier"]

    assert "scaler" in preprocessing.named_steps
    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight == "balanced"
    assert classifier.solver == "liblinear"
    assert classifier.l1_ratio == 0.0
    assert classifier.max_iter == 2_000
    assert classifier.random_state == 123


def test_logistic_specification_can_disable_class_weighting() -> None:
    """The sensitivity model must permit unweighted fitting."""

    specification = build_logistic_specification(
        class_weighting=False,
    )

    classifier = specification.estimator.named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight is None


def test_logistic_grid_contains_prespecified_c_values() -> None:
    """The inner-validation grid must target regularization."""

    specification = build_logistic_specification()

    assert specification.parameter_grid == {
        "classifier__C": LOGISTIC_C_VALUES,
    }
    assert LOGISTIC_C_VALUES == (
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    )


def test_logistic_pipeline_produces_valid_probabilities() -> None:
    """The full logistic pipeline must fit with missing values."""

    predictors, outcome = make_classification_data()
    specification = build_logistic_specification()

    specification.estimator.fit(predictors, outcome)
    probabilities = specification.estimator.predict_proba(predictors)

    assert probabilities.shape == (len(predictors), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_xgboost_specification_is_weighted_and_unscaled_by_default() -> None:
    """The primary XGBoost model must use local weighting without scaling."""

    specification = build_xgboost_specification(
        random_state=127,
        n_jobs=2,
    )

    preprocessing = specification.estimator.named_steps["preprocessing"]
    classifier = specification.estimator.named_steps["classifier"]

    assert "scaler" not in preprocessing.named_steps
    assert isinstance(classifier, BalancedXGBClassifier)
    assert classifier.class_weighting is True
    assert classifier.subsample == 0.8
    assert classifier.colsample_bytree == 0.8
    assert classifier.min_child_weight == 1.0
    assert classifier.reg_lambda == 1.0
    assert classifier.random_state == 127
    assert classifier.n_jobs == 2


def test_xgboost_specification_can_disable_class_weighting() -> None:
    """The sensitivity model must permit unweighted XGBoost fitting."""

    specification = build_xgboost_specification(
        class_weighting=False,
    )

    classifier = specification.estimator.named_steps["classifier"]

    assert isinstance(classifier, BalancedXGBClassifier)
    assert classifier.class_weighting is False


def test_xgboost_grid_contains_eight_candidates() -> None:
    """The compact XGBoost grid must contain eight candidates."""

    specification = build_xgboost_specification()

    assert specification.parameter_grid == {
        "classifier__n_estimators": XGBOOST_N_ESTIMATORS,
        "classifier__max_depth": XGBOOST_MAX_DEPTHS,
        "classifier__learning_rate": XGBOOST_LEARNING_RATES,
    }
    assert XGBOOST_N_ESTIMATORS == (100, 300)
    assert XGBOOST_MAX_DEPTHS == (2, 3)
    assert XGBOOST_LEARNING_RATES == (0.03, 0.1)

    candidate_count = (
        len(XGBOOST_N_ESTIMATORS)
        * len(XGBOOST_MAX_DEPTHS)
        * len(XGBOOST_LEARNING_RATES)
    )

    assert candidate_count == 8


def test_model_registry_propagates_disabled_class_weighting() -> None:
    """The registry must pass the sensitivity setting to both models."""

    specifications = build_model_specifications(
        class_weighting=False,
    )

    logistic_classifier = specifications["logistic_regression"].estimator.named_steps[
        "classifier"
    ]

    xgboost_classifier = specifications["xgboost"].estimator.named_steps["classifier"]

    assert logistic_classifier.class_weight is None
    assert xgboost_classifier.class_weighting is False


def test_model_registry_propagates_disabled_missing_indicators() -> None:
    """Every model must receive the missing-indicator setting."""

    specifications = build_model_specifications(
        add_missing_indicators=False,
    )

    for specification in specifications.values():
        preprocessing = specification.estimator.named_steps["preprocessing"]
        imputer = preprocessing.named_steps["imputer"]

        assert imputer.add_indicator is False
