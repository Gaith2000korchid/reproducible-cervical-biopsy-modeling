"""Leakage-safe preprocessing fitted exclusively on training data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

DEFAULT_MAX_MISSING_FRACTION = 0.80


class MissingnessFilter(TransformerMixin, BaseEstimator):
    """Remove features exceeding a training-fold missingness threshold."""

    def __init__(
        self,
        max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
    ) -> None:
        self.max_missing_fraction = max_missing_fraction

    @staticmethod
    def _require_dataframe(data: Any) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("MissingnessFilter requires a pandas DataFrame.")
        return data

    def fit(
        self,
        data: pd.DataFrame,
        outcome: pd.Series | None = None,
    ) -> MissingnessFilter:
        """Learn retained feature names from training data only."""
        del outcome
        data = self._require_dataframe(data)

        if not 0.0 <= self.max_missing_fraction <= 1.0:
            raise ValueError("max_missing_fraction must be between 0 and 1.")

        if data.shape[1] == 0:
            raise ValueError("At least one predictor is required.")

        self.n_features_in_ = data.shape[1]
        self.feature_names_in_ = np.asarray(data.columns, dtype=object)
        self.missing_fractions_ = data.isna().mean()

        retained = self.missing_fractions_ <= self.max_missing_fraction
        self.selected_features_ = np.asarray(
            self.missing_fractions_.index[retained],
            dtype=object,
        )

        if self.selected_features_.size == 0:
            raise ValueError("No predictor satisfies the missingness threshold.")

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retain only features selected during training."""
        check_is_fitted(self, "selected_features_")
        data = self._require_dataframe(data)

        observed_features = np.asarray(data.columns, dtype=object)
        if not np.array_equal(
            observed_features,
            self.feature_names_in_,
        ):
            raise ValueError(
                "Transform columns must match the fitted columns and order."
            )

        return data.loc[:, self.selected_features_].copy()

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """Return retained feature names."""
        check_is_fitted(self, "selected_features_")

        if input_features is not None:
            provided_features = np.asarray(input_features, dtype=object)
            if not np.array_equal(
                provided_features,
                self.feature_names_in_,
            ):
                raise ValueError("input_features must match fitted feature names.")

        return self.selected_features_.copy()


def build_preprocessing_pipeline(
    *,
    scale: bool,
    max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
    add_missing_indicators: bool = True,
) -> Pipeline:
    """Build median imputation, indicators, filtering, and optional scaling."""
    steps: list[tuple[str, Any]] = [
        (
            "missingness_filter",
            MissingnessFilter(
                max_missing_fraction=max_missing_fraction,
            ),
        ),
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=add_missing_indicators,
                keep_empty_features=False,
            ),
        ),
        (
            "constant_filter",
            VarianceThreshold(threshold=0.0),
        ),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps)
    pipeline.set_output(transform="pandas")
    return pipeline
