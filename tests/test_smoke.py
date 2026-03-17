from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from puate.data import generate_case_control_synthetic_linear, generate_censoring_synthetic_linear
from puate.estimators import estimate_case_control_ate, estimate_censoring_ate
from puate.models import BoundedProbClassifier, LinearPULearner
from puate.reporting import summarize_trials


def test_censoring_smoke() -> None:
    data = generate_censoring_synthetic_linear(n=200, p=3, rng=np.random.default_rng(0))
    result = estimate_censoring_ate(
        data.X,
        data.O,
        data.Y,
        BoundedProbClassifier(LogisticRegression(max_iter=1000)),
        LinearRegression(),
        LinearRegression(),
        float(data.metadata["observation_rate"]),
        random_state=0,
        true_g=data.true_propensity,
    )
    assert set(result.estimates.keys()) == {"ipw", "dm", "efficient", "ipw_true_g", "dm_true_g", "efficient_true_g"}


def test_case_control_smoke() -> None:
    data = generate_case_control_synthetic_linear(
        n_positive=80,
        n_unlabeled=160,
        p=3,
        class_prior=0.3,
        rng=np.random.default_rng(0),
    )
    result = estimate_case_control_ate(
        data.X,
        data.O,
        data.Y,
        BoundedProbClassifier(LinearPULearner(prior=0.3)),
        LinearRegression(),
        LinearRegression(),
        class_prior=0.3,
        random_state=0,
        true_e=data.true_propensity,
    )
    assert set(result.estimates.keys()) == {"ipw", "dm", "efficient", "ipw_true_e", "dm_true_e", "efficient_true_e"}


def test_summary_smoke() -> None:
    import pandas as pd

    estimates = pd.DataFrame({"ipw": [2.9, 3.1], "dm": [3.0, 3.0]})
    variances = pd.DataFrame({"ipw": [1.0, 1.0], "dm": [0.5, 0.5]})
    summary = summarize_trials(estimates, variances, true_ate=3.0, ci_sample_size=100)
    assert list(summary.index) == ["MSE", "Bias", "Cov. ratio"]
