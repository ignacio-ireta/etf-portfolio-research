"""Model registry for ETF ML research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from etf_portfolio.config import MLTask


@dataclass
class HistoricalMeanRegressor(BaseEstimator):
    mean_: float = 0.0

    def fit(self, X, y):  # noqa: N803
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X):  # noqa: N803
        return np.full(len(X), self.mean_, dtype=float)


@dataclass
class HistoricalMeanClassifier(BaseEstimator):
    positive_rate_: float = 0.5

    def fit(self, X, y):  # noqa: N803
        self.positive_rate_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X):  # noqa: N803
        label = 1 if self.positive_rate_ >= 0.5 else 0
        return np.full(len(X), label, dtype=int)

    def predict_proba(self, X):  # noqa: N803
        negative = 1.0 - self.positive_rate_
        return np.column_stack(
            [
                np.full(len(X), negative, dtype=float),
                np.full(len(X), self.positive_rate_, dtype=float),
            ]
        )


def build_model(
    model_name: str,
    *,
    task: MLTask,
    random_state: int = 42,
) -> BaseEstimator:
    """Instantiate a supported model for the configured ML task."""

    if model_name == "historical_mean":
        return HistoricalMeanClassifier() if task == "classification" else HistoricalMeanRegressor()

    if model_name == "ridge":
        if task == "classification":
            return LogisticRegression(
                C=1.0,
                max_iter=2_000,
                random_state=random_state,
            )
        return Ridge(alpha=1.0, random_state=random_state)

    if model_name == "random_forest":
        if task == "classification":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=10,
                random_state=random_state,
            )
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported ML model: {model_name}.")


__all__ = [
    "HistoricalMeanClassifier",
    "HistoricalMeanRegressor",
    "build_model",
]
