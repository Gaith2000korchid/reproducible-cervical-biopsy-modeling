"""XGBoost classifier with optional training-local class weighting."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

DEFAULT_RANDOM_STATE = 42


class BalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost classifier with configurable training-local weighting."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        class_weighting: bool = True,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.class_weighting = class_weighting
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        predictors: Any,
        outcome: Any,
    ) -> BalancedXGBClassifier:
        """Fit XGBoost using optional training-local class weighting."""

        from xgboost import XGBClassifier

        outcome_array = np.asarray(outcome)
        classes, counts = np.unique(
            outcome_array,
            return_counts=True,
        )

        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(
                "BalancedXGBClassifier requires both binary classes encoded as 0 and 1."
            )

        negative_count = int(counts[0])
        positive_count = int(counts[1])

        if self.class_weighting:
            self.scale_pos_weight_ = negative_count / positive_count
        else:
            self.scale_pos_weight_ = 1.0

        self.estimator_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight_,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.estimator_.fit(
            predictors,
            outcome_array,
        )

        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        return self

    def predict(self, predictors: Any) -> np.ndarray:
        """Predict binary class labels."""

        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(predictors)

    def predict_proba(self, predictors: Any) -> np.ndarray:
        """Predict class probabilities."""

        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(predictors)
