
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BoundedProbClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier and clip ``predict_proba`` outputs away from 0 and 1.

    The original notebooks repeatedly clipped probabilities to avoid numerical
    instability in inverse-propensity weights. This wrapper centralizes that
    behavior without changing the underlying estimator.
    """

    def __init__(self, base_classifier: BaseEstimator, lower: float = 0.05, upper: float = 0.95):
        self.base_classifier = base_classifier
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.base_classifier.predict_proba(X)
        pos = np.clip(proba[:, 1], self.lower, self.upper)
        return np.column_stack([1.0 - pos, pos])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.base_classifier.score(X, y)


class ElkanNotoClassifier(BaseEstimator, ClassifierMixin):
    """Elkan-Noto style rescaling for the positive class probability.

    The original notebooks divided the *entire* 2-column probability matrix by
    the scaling factor and only used the positive-class column. This public
    version keeps the intended behavior for the positive class while returning a
    valid 2-column probability matrix whose rows sum to one.
    """

    def __init__(
        self,
        base_classifier: BaseEstimator,
        scale_factor: float = 1.0,
        lower: float = 0.10,
        upper: float = 0.90,
    ):
        self.base_classifier = base_classifier
        self.scale_factor = scale_factor
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base_classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.base_classifier.predict_proba(X)
        pos = np.clip(proba[:, 1] / self.scale_factor, self.lower, self.upper)
        return np.column_stack([1.0 - pos, pos])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.base_classifier.score(X, y)


class NonNegativePULearner(BaseEstimator, ClassifierMixin):
    """Non-negative PU learner used in the case-control MLP notebooks.

    This class mirrors the provided PyTorch implementation as closely as
    possible so that the cleaned scripts behave like the original notebooks.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        prior: float | None = None,
        lr: float = 0.01,
        epochs: int = 100,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.prior = prior
        self.lr = lr
        self.epochs = epochs
        self.is_fitted_ = False

    def _build_model(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)

        x_tensor = torch.FloatTensor(X)
        pos_mask = y == 1
        unlabeled_mask = y == 0

        x_pos = x_tensor[pos_mask]
        x_unlabeled = x_tensor[unlabeled_mask]

        if self.prior is None:
            self.prior = float(np.mean(pos_mask))

        self.model_ = self._build_model(X.shape[1])
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pos_preds = torch.clamp(self.model_(x_pos), min=0.05, max=0.95).squeeze()
            unlabeled_preds = torch.clamp(self.model_(x_unlabeled), min=0.05, max=0.95).squeeze()

            pos_loss = -torch.mean(torch.log(pos_preds))
            pos_loss2 = -torch.mean(torch.log(1.0 - pos_preds))
            unlabeled_loss = -torch.mean(torch.log(1.0 - unlabeled_preds))
            loss_p = self.prior * pos_loss
            loss_n = unlabeled_loss - self.prior * pos_loss2

            if loss_n > 0:
                loss = loss_p + loss_n
            else:
                loss = -0.01 * loss_n

            loss.backward()
            optimizer.step()

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["is_fitted_", "model_"])
        X = check_array(X)
        x_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            pos = self.model_(x_tensor).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y, self.predict(X)))


class LinearPULearner(BaseEstimator, ClassifierMixin):
    """Linear PU learner from the provided case-control linear notebook."""

    def __init__(self, prior: float | None = None):
        self.prior = prior
        self.is_fitted_ = False

    def _pu_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, prior: float) -> float:
        logits = X @ w
        pos_mask = y == 1
        unlabeled_mask = y == 0
        return float(
            -prior * np.mean(logits[pos_mask])
            + np.mean(np.logaddexp(0.0, logits[unlabeled_mask]))
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)

        if self.prior is None:
            self.prior = float(np.mean(y == 1))

        n_features = X.shape[1]
        w0 = np.ones(n_features)
        res = minimize(self._pu_loss, w0, args=(X, y, self.prior), method="BFGS")
        self.coef_ = res.x
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=["is_fitted_", "coef_"])
        X = check_array(X)
        logits = X @ self.coef_
        pos = expit(logits)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y, self.predict(X)))
