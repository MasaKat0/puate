
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal


@dataclass
class ExperimentData:
    X: np.ndarray
    O: np.ndarray
    Y: np.ndarray
    true_ate: float
    metadata: dict[str, float | int | str]
    true_propensity: np.ndarray | None = None


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def generate_censoring_synthetic_linear(
    *,
    n: int = 3000,
    p: int = 3,
    true_ate: float = 3.0,
    rng: np.random.Generator | None = None,
) -> ExperimentData:
    rng = _rng(rng)
    X = rng.normal(0.0, 1.0, size=(n, p))
    propensity_coef = rng.normal(0.0, 0.5, size=p)
    eta = expit(X @ propensity_coef)
    eta = np.clip(eta, 0.10, 0.90)
    D = rng.binomial(1, eta)
    observation_rate = float(rng.uniform(0.1, 0.9))
    O = rng.binomial(1, observation_rate, size=n) * D
    beta = rng.normal(0.0, 1.0, size=p)
    Y = X @ beta + 1.1 + true_ate * D + rng.normal(0.0, 1.0, size=n)
    g = ((1.0 - observation_rate) * eta) / (1.0 - observation_rate * eta)
    g = np.clip(g, 0.10, 0.90)
    return ExperimentData(
        X=X,
        O=O.astype(int),
        Y=Y,
        true_ate=true_ate,
        true_propensity=g,
        metadata={
            "setting": "censoring",
            "design": "synthetic_linear",
            "sample_size": n,
            "p": p,
            "observation_rate": observation_rate,
        },
    )


def generate_censoring_synthetic_nonlinear(
    *,
    n: int = 3000,
    p: int = 10,
    true_ate: float = 3.0,
    rng: np.random.Generator | None = None,
) -> ExperimentData:
    rng = _rng(rng)
    X = rng.normal(0.0, 1.0, size=(n, p))
    X_aug = np.concatenate([X, X**2], axis=1)
    propensity_coef = rng.normal(0.0, 0.5, size=X_aug.shape[1])
    eta = expit(X_aug @ propensity_coef)
    eta = np.clip(eta, 0.10, 0.90)
    D = rng.binomial(1, eta)
    observation_rate = float(rng.uniform(0.1, 0.9))
    O = rng.binomial(1, observation_rate, size=n) * D
    beta = rng.normal(0.0, 1.0, size=p)
    Y = (X @ beta) ** 2 + 1.1 + true_ate * D + rng.normal(0.0, 1.0, size=n)
    g = ((1.0 - observation_rate) * eta) / (1.0 - observation_rate * eta)
    g = np.clip(g, 0.10, 0.90)
    return ExperimentData(
        X=X,
        O=O.astype(int),
        Y=Y,
        true_ate=true_ate,
        true_propensity=g,
        metadata={
            "setting": "censoring",
            "design": "synthetic_nonlinear",
            "sample_size": n,
            "p": p,
            "observation_rate": observation_rate,
        },
    )


def _case_control_common(
    *,
    n_positive: int,
    n_unlabeled: int,
    p: int,
    class_prior: float,
    nonlinear: bool,
    true_ate: float,
    rng: np.random.Generator,
) -> ExperimentData:
    mu_p = 0.5
    mu_n = 0.0
    n_unlabeled_positive = int(n_unlabeled * class_prior)
    n_unlabeled_negative = n_unlabeled - n_unlabeled_positive

    X_p = rng.normal(mu_p, 1.0, size=(n_positive, p))
    X_p_u = rng.normal(mu_p, 1.0, size=(n_unlabeled_positive, p))
    X_n_u = rng.normal(mu_n, 1.0, size=(n_unlabeled_negative, p))

    X = np.concatenate([X_p, X_p_u, X_n_u], axis=0)
    O = np.zeros(n_positive + n_unlabeled, dtype=int)
    O[:n_positive] = 1

    ppx = multivariate_normal.pdf(X, mean=np.ones(p) * mu_p, cov=np.ones(p))
    pnx = multivariate_normal.pdf(X, mean=np.ones(p) * mu_n, cov=np.ones(p))
    pux = class_prior * ppx + (1.0 - class_prior) * pnx
    e = np.clip(class_prior * ppx / pux, 0.10, 0.90)

    D = np.zeros(n_positive + n_unlabeled, dtype=int)
    D[: n_positive + n_unlabeled_positive] = 1

    beta = rng.normal(0.0, 1.0, size=p)
    if nonlinear:
        Y = (X @ beta) ** 2 + 1.1 + true_ate * D + rng.normal(0.0, 1.0, size=len(D))
        design = "synthetic_nonlinear"
    else:
        Y = X @ beta + 1.1 + true_ate * D + rng.normal(0.0, 1.0, size=len(D))
        design = "synthetic_linear"

    return ExperimentData(
        X=X,
        O=O,
        Y=Y,
        true_ate=true_ate,
        true_propensity=e,
        metadata={
            "setting": "case_control",
            "design": design,
            "n_positive": n_positive,
            "n_unlabeled": n_unlabeled,
            "p": p,
            "class_prior": class_prior,
        },
    )


def generate_case_control_synthetic_linear(
    *,
    n_positive: int = 1000,
    n_unlabeled: int = 2000,
    p: int = 3,
    class_prior: float = 0.3,
    true_ate: float = 3.0,
    rng: np.random.Generator | None = None,
) -> ExperimentData:
    return _case_control_common(
        n_positive=n_positive,
        n_unlabeled=n_unlabeled,
        p=p,
        class_prior=class_prior,
        nonlinear=False,
        true_ate=true_ate,
        rng=_rng(rng),
    )


def generate_case_control_synthetic_nonlinear(
    *,
    n_positive: int = 1000,
    n_unlabeled: int = 2000,
    p: int = 3,
    class_prior: float = 0.3,
    true_ate: float = 3.0,
    rng: np.random.Generator | None = None,
) -> ExperimentData:
    return _case_control_common(
        n_positive=n_positive,
        n_unlabeled=n_unlabeled,
        p=p,
        class_prior=class_prior,
        nonlinear=True,
        true_ate=true_ate,
        rng=_rng(rng),
    )


def _load_ihdp_surface(surface: str):
    try:
        from econml.data import dgps
    except ImportError as exc:
        raise ImportError(
            "IHDP experiments require econml. Install the optional dependency "
            "with `pip install -e \".[ihdp]\"` or `pip install econml`."
        ) from exc

    surface = surface.upper()
    if surface == "A":
        return dgps.ihdp_surface_A()
    if surface == "B":
        return dgps.ihdp_surface_B()
    raise ValueError("surface must be 'A' or 'B'")


def make_censoring_ihdp_observations(
    *,
    surface: str = "A",
    observation_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> ExperimentData:
    """Mirror the IHDP censoring notebooks with the IHDP-B surface bug fixed."""

    rng = _rng(rng)
    Y, D, X, true_ite = _load_ihdp_surface(surface)
    O = rng.binomial(1, observation_rate, size=len(D)) * D
    return ExperimentData(
        X=np.asarray(X),
        O=np.asarray(O, dtype=int),
        Y=np.asarray(Y, dtype=float),
        true_ate=float(np.mean(true_ite)),
        true_propensity=None,
        metadata={
            "setting": "censoring",
            "design": f"ihdp_surface_{surface.upper()}",
            "sample_size": int(len(D)),
            "observation_rate": float(observation_rate),
            "surface": surface.upper(),
        },
    )


def make_case_control_ihdp_observations(
    *,
    surface: str = "A",
    class_prior: float = 0.1,
    rng: np.random.Generator | None = None,
) -> ExperimentData:
    """Mirror the provided IHDP case-control notebooks.

    Surface A and surface B used different sampling code in the original
    notebooks. This helper preserves that behavior to avoid silently changing
    the experimental design in the public refactor.
    """

    rng = _rng(rng)
    Y, D, X, true_ite = _load_ihdp_surface(surface)
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=int)
    surface = surface.upper()

    if surface == "A":
        true_positive_size = int(np.sum(D))
        true_negative_size = int(np.sum(1 - D))
        unlabeled_positive_size = int(class_prior * true_negative_size / (1.0 - class_prior))
        labeled_positive_size = true_positive_size - unlabeled_positive_size

        positive_index = np.where(D == 1)[0]
        positive_index = rng.permutation(positive_index)
        labeled_positive_index = positive_index[:labeled_positive_size]

        O = np.zeros(len(D), dtype=int)
        O[labeled_positive_index] = 1
        X_out = X
        Y_out = Y
        O_out = O
    else:
        unlabeled_size = int(len(D) / 2)
        sample_index = rng.permutation(np.arange(len(X)))
        unlabeled_sample_index = sample_index[:unlabeled_size]
        positive_sample_index = sample_index[unlabeled_size:]
        positive_label_index = positive_sample_index[D[positive_sample_index] == 1]

        total_index = np.concatenate([positive_label_index, unlabeled_sample_index])
        O = np.zeros(len(D), dtype=int)
        O[positive_label_index] = 1

        X_out = X[total_index]
        Y_out = Y[total_index]
        O_out = O[total_index]

    return ExperimentData(
        X=np.asarray(X_out),
        O=np.asarray(O_out, dtype=int),
        Y=np.asarray(Y_out, dtype=float),
        true_ate=float(np.mean(true_ite)),
        true_propensity=None,
        metadata={
            "setting": "case_control",
            "design": f"ihdp_surface_{surface}",
            "sample_size": int(len(X_out)),
            "class_prior": float(class_prior),
            "surface": surface,
        },
    )
