
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from puate.data import (
    generate_case_control_synthetic_linear,
    generate_censoring_synthetic_linear,
)
from puate.estimators import estimate_case_control_ate, estimate_censoring_ate
from puate.models import BoundedProbClassifier, LinearPULearner


def test_censoring_smoke() -> None:
    data = generate_censoring_synthetic_linear(n=200, p=3, rng=np.random.default_rng(0))
    result = estimate_censoring_ate(
        data.X,
        data.O,
        data.Y,
        observation_model=BoundedProbClassifier(LogisticRegression(max_iter=1000)),
        mu_t_model=LinearRegression(),
        mu_u_model=LinearRegression(),
        observation_rate=float(data.metadata["observation_rate"]),
        n_folds=2,
        random_state=0,
        true_g=data.true_propensity,
    )
    assert set(result.estimates) == {
        "ipw",
        "dm",
        "efficient",
        "ipw_true_g",
        "dm_true_g",
        "efficient_true_g",
    }
    assert all(np.isfinite(v) for v in result.estimates.values())
    assert all(v >= 0.0 for v in result.asymptotic_variances.values())


def test_case_control_smoke() -> None:
    data = generate_case_control_synthetic_linear(
        n_positive=100,
        n_unlabeled=200,
        p=3,
        rng=np.random.default_rng(1),
    )
    result = estimate_case_control_ate(
        data.X,
        data.O,
        data.Y,
        propensity_model=BoundedProbClassifier(LinearPULearner(prior=float(data.metadata["class_prior"]))),
        mu_t_model=LinearRegression(),
        mu_u_model=LinearRegression(),
        class_prior=float(data.metadata["class_prior"]),
        n_folds=2,
        random_state=0,
        true_e=data.true_propensity,
    )
    assert set(result.estimates) == {
        "ipw",
        "dm",
        "efficient",
        "ipw_true_e",
        "dm_true_e",
        "efficient_true_e",
    }
    assert all(np.isfinite(v) for v in result.estimates.values())
    assert all(v >= 0.0 for v in result.asymptotic_variances.values())
