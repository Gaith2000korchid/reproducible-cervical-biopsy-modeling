"""Tests for the training-weighted XGBoost wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from cervical_biopsy_modeling.xgboost_model import (
    BalancedXGBClassifier,
)


def make_predictors() -> pd.DataFrame:
    """Create a small numeric classification dataset."""
    return pd.DataFrame(
        {
            "first": np.linspace(0.0, 1.0, 12),
            "second": [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
        }
    )


def build_small_classifier() -> BalancedXGBClassifier:
    """Build a fast classifier for unit tests."""
    return BalancedXGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        random_state=17,
        n_jobs=1,
    )


def test_xgboost_wrapper_is_cloneable() -> None:
    """The wrapper must satisfy scikit-learn cloning."""
    classifier = build_small_classifier()
    cloned = clone(classifier)

    assert isinstance(cloned, BalancedXGBClassifier)
    assert cloned.get_params() == classifier.get_params()
    assert cloned is not classifier


def test_xgboost_wrapper_requires_both_classes() -> None:
    """Training data must contain classes zero and one."""
    predictors = make_predictors()
    outcome = np.zeros(len(predictors), dtype=int)
    classifier = build_small_classifier()

    with pytest.raises(ValueError, match="both binary classes"):
        classifier.fit(predictors, outcome)


def test_xgboost_wrapper_computes_training_local_weight() -> None:
    """The imbalance ratio must be derived during fit."""
    predictors = make_predictors()
    outcome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    classifier = build_small_classifier()

    classifier.fit(predictors, outcome)

    assert classifier.scale_pos_weight_ == pytest.approx(3.0)
    assert classifier.estimator_.get_params()["scale_pos_weight"] == pytest.approx(3.0)


def test_xgboost_wrapper_updates_weight_on_refit() -> None:
    """A second fit must recompute rather than reuse the weight."""
    predictors = make_predictors()
    first_outcome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    second_outcome = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    classifier = build_small_classifier()

    classifier.fit(predictors, first_outcome)
    assert classifier.scale_pos_weight_ == pytest.approx(3.0)

    classifier.fit(predictors, second_outcome)
    assert classifier.scale_pos_weight_ == pytest.approx(1.0)


def test_xgboost_wrapper_produces_valid_probabilities() -> None:
    """Fitted probabilities must have the binary shape."""
    predictors = make_predictors()
    outcome = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    classifier = build_small_classifier()

    classifier.fit(predictors, outcome)
    probabilities = classifier.predict_proba(predictors)

    assert probabilities.shape == (len(predictors), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_xgboost_wrapper_rejects_prediction_before_fit() -> None:
    """Prediction before fitting must raise a clear error."""
    predictors = make_predictors()
    classifier = build_small_classifier()

    with pytest.raises(NotFittedError):
        classifier.predict_proba(predictors)
