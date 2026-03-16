
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.base import clone


@dataclass
class ATEEstimate:
    """Container for repeated experiment outputs."""

    estimates: "OrderedDict[str, float]"
    asymptotic_variances: "OrderedDict[str, float]"

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "estimates": dict(self.estimates),
            "asymptotic_variances": dict(self.asymptotic_variances),
        }


def _clip_probs(p: np.ndarray, lower: float = 0.10, upper: float = 0.90) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), lower, upper)


def _make_folds(n: int, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(indices, n_folds)]


def censoring_propensity_from_observation_prob(
    observation_prob: np.ndarray,
    observation_rate: float,
    lower: float = 0.10,
    upper: float = 0.90,
) -> np.ndarray:
    """Convert ``P(O=1|X)`` to ``g(1|X)=P(D=1|X,O=0)``.

    In the original notebooks the scalar ``prior`` in the censoring setting is
    actually the observation rate :math:`c = P(O=1 \\mid D=1)`, not the class
    prior. This helper keeps the original formula but names it explicitly.
    """

    observation_prob = _clip_probs(observation_prob, lower, upper)
    g = observation_prob * (1.0 - observation_rate) / ((1.0 - observation_prob) * observation_rate)
    return _clip_probs(g, lower, upper)


def _mean_and_var(scores: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(scores))
    var = float(np.mean((scores - mean) ** 2))
    return mean, var


def estimate_censoring_ate(
    X: np.ndarray,
    O: np.ndarray,
    Y: np.ndarray,
    observation_model,
    mu_t_model,
    mu_u_model,
    observation_rate: float,
    *,
    n_folds: int = 2,
    random_state: int | None = None,
    true_g: np.ndarray | None = None,
) -> ATEEstimate:
    """Estimate ATE in the censoring setting.

    Parameters
    ----------
    X, O, Y:
        Covariates, observation indicator, and outcome.
    observation_model:
        Classifier estimating ``P(O=1|X)``.
    mu_t_model, mu_u_model:
        Regression models for ``E[Y|X,O=1]`` and ``E[Y|X,O=0]``.
    observation_rate:
        Probability that a treated unit is observed in the positive set
        (``prior`` in the original notebooks).
    true_g:
        Optional ground-truth censoring propensity ``g(1|X)`` used in the
        synthetic experiments.
    """

    X = np.asarray(X)
    O = np.asarray(O).astype(int)
    Y = np.asarray(Y).astype(float)

    if set(np.unique(O)) != {0, 1}:
        raise ValueError("O must contain both 0 and 1 in the full sample.")

    rng = np.random.default_rng(random_state)
    folds = _make_folds(len(X), n_folds, rng)

    estimated_scores: OrderedDict[str, list[np.ndarray]] = OrderedDict(
        [("ipw", []), ("dm", []), ("efficient", [])]
    )
    oracle_scores: OrderedDict[str, list[np.ndarray]] | None = None
    if true_g is not None:
        true_g = np.asarray(true_g, dtype=float)
        oracle_scores = OrderedDict(
            [("ipw_true_g", []), ("dm_true_g", []), ("efficient_true_g", [])]
        )

    for test_idx in folds:
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        X_train, O_train, Y_train = X[train_idx], O[train_idx], Y[train_idx]
        X_test, O_test, Y_test = X[test_idx], O[test_idx], Y[test_idx]

        obs_model = clone(observation_model)
        mu_t = clone(mu_t_model)
        mu_u = clone(mu_u_model)

        obs_model.fit(X_train, O_train)
        pi_hat = _clip_probs(obs_model.predict_proba(X_test)[:, 1], 0.05, 0.95)
        g_hat = censoring_propensity_from_observation_prob(pi_hat, observation_rate)

        mu_t.fit(X_train[O_train == 1], Y_train[O_train == 1])
        mu_t_hat = np.asarray(mu_t.predict(X_test), dtype=float)

        mu_u.fit(X_train[O_train == 0], Y_train[O_train == 0])
        mu_u_hat = np.asarray(mu_u.predict(X_test), dtype=float)

        ipw = (
            O_test * Y_test / pi_hat
            - (1 - O_test) * Y_test / ((1.0 - g_hat) * (1.0 - pi_hat))
            + g_hat * O_test * Y_test / ((1.0 - g_hat) * pi_hat)
        )
        dm = mu_t_hat - mu_u_hat / (1.0 - g_hat) + g_hat * mu_t_hat / (1.0 - g_hat)
        efficient = (
            O_test * (Y_test - mu_t_hat) / pi_hat
            - (1 - O_test) * (Y_test - mu_u_hat) / ((1.0 - g_hat) * (1.0 - pi_hat))
            + g_hat * O_test * (Y_test - mu_t_hat) / ((1.0 - g_hat) * pi_hat)
            + dm
        )

        estimated_scores["ipw"].append(ipw)
        estimated_scores["dm"].append(dm)
        estimated_scores["efficient"].append(efficient)

        if oracle_scores is not None:
            g_star = _clip_probs(true_g[test_idx], 0.10, 0.90)
            ipw_true = (
                O_test * Y_test / pi_hat
                - (1 - O_test) * Y_test / ((1.0 - g_star) * (1.0 - pi_hat))
                + g_star * O_test * Y_test / ((1.0 - g_star) * pi_hat)
            )
            dm_true = mu_t_hat - mu_u_hat / (1.0 - g_star) + g_star * mu_t_hat / (1.0 - g_star)
            efficient_true = (
                O_test * (Y_test - mu_t_hat) / pi_hat
                - (1 - O_test) * (Y_test - mu_u_hat) / ((1.0 - g_star) * (1.0 - pi_hat))
                + g_star * O_test * (Y_test - mu_t_hat) / ((1.0 - g_star) * pi_hat)
                + dm_true
            )
            oracle_scores["ipw_true_g"].append(ipw_true)
            oracle_scores["dm_true_g"].append(dm_true)
            oracle_scores["efficient_true_g"].append(efficient_true)

    estimates: OrderedDict[str, float] = OrderedDict()
    variances: OrderedDict[str, float] = OrderedDict()
    for name, chunks in estimated_scores.items():
        score = np.concatenate(chunks)
        estimates[name], variances[name] = _mean_and_var(score)

    if oracle_scores is not None:
        for name, chunks in oracle_scores.items():
            score = np.concatenate(chunks)
            estimates[name], variances[name] = _mean_and_var(score)

    return ATEEstimate(estimates=estimates, asymptotic_variances=variances)


def _case_control_group_stats(
    treatment_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
) -> tuple[float, float]:
    n_t = len(treatment_scores)
    n_u = len(unlabeled_scores)
    n = n_t + n_u
    alpha = n_t / n
    estimate = float(np.mean(treatment_scores) + np.mean(unlabeled_scores))
    asymptotic_variance = float(
        np.var(treatment_scores, ddof=0) / alpha
        + np.var(unlabeled_scores, ddof=0) / (1.0 - alpha)
    )
    return estimate, asymptotic_variance


def estimate_case_control_ate(
    X: np.ndarray,
    O: np.ndarray,
    Y: np.ndarray,
    propensity_model,
    mu_t_model,
    mu_u_model,
    class_prior: float,
    *,
    n_folds: int = 2,
    random_state: int | None = None,
    true_e: np.ndarray | None = None,
) -> ATEEstimate:
    """Estimate ATE in the case-control setting.

    Notes
    -----
    The original notebooks contained two clear issues in the case-control
    variance code:

    1. the DM variance used the DR treatment residual array by mistake, and
    2. the groupwise variance calculation mixed raw second moments and
       incompatible score indices.

    This function keeps the point estimators intact but computes the asymptotic
    variances from the positive and unlabeled groups separately, which matches
    the two-sample estimator structure.
    """

    X = np.asarray(X)
    O = np.asarray(O).astype(int)
    Y = np.asarray(Y).astype(float)

    if set(np.unique(O)) != {0, 1}:
        raise ValueError("O must contain both 0 (unlabeled) and 1 (positive sample).")

    rng = np.random.default_rng(random_state)
    folds = _make_folds(len(X), n_folds, rng)

    estimated: dict[str, list[np.ndarray]] = {
        "ipw_t": [],
        "ipw_u": [],
        "dm_u": [],
        "efficient_t": [],
        "efficient_u": [],
    }
    oracle: dict[str, list[np.ndarray]] | None = None
    if true_e is not None:
        true_e = np.asarray(true_e, dtype=float)
        oracle = {
            "ipw_t_true_e": [],
            "ipw_u_true_e": [],
            "dm_u_true_e": [],
            "efficient_t_true_e": [],
            "efficient_u_true_e": [],
        }

    for test_idx in folds:
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        X_train, O_train, Y_train = X[train_idx], O[train_idx], Y[train_idx]
        X_test, O_test, Y_test = X[test_idx], O[test_idx], Y[test_idx]

        prop_model = clone(propensity_model)
        mu_t = clone(mu_t_model)
        mu_u = clone(mu_u_model)

        prop_model.fit(X_train, O_train)
        e_hat = _clip_probs(prop_model.predict_proba(X_test)[:, 1], 0.05, 0.95)

        mu_t.fit(X_train[O_train == 1], Y_train[O_train == 1])
        mu_t_hat = np.asarray(mu_t.predict(X_test), dtype=float)

        mu_u.fit(X_train[O_train == 0], Y_train[O_train == 0])
        mu_u_hat = np.asarray(mu_u.predict(X_test), dtype=float)

        t_mask = O_test == 1
        u_mask = ~t_mask

        ipw_t = class_prior * Y_test[t_mask] / (e_hat[t_mask] * (1.0 - e_hat[t_mask]))
        ipw_u = -Y_test[u_mask] / (1.0 - e_hat[u_mask])

        dm_u = (
            mu_t_hat[u_mask]
            - mu_u_hat[u_mask] / (1.0 - e_hat[u_mask])
            + e_hat[u_mask] * mu_t_hat[u_mask] / (1.0 - e_hat[u_mask])
        )

        efficient_t = (
            class_prior
            * (Y_test[t_mask] - mu_t_hat[t_mask])
            / (e_hat[t_mask] * (1.0 - e_hat[t_mask]))
        )
        efficient_u = (
            -(Y_test[u_mask] - mu_u_hat[u_mask]) / (1.0 - e_hat[u_mask]) + dm_u
        )

        estimated["ipw_t"].append(ipw_t)
        estimated["ipw_u"].append(ipw_u)
        estimated["dm_u"].append(dm_u)
        estimated["efficient_t"].append(efficient_t)
        estimated["efficient_u"].append(efficient_u)

        if oracle is not None:
            e_star = _clip_probs(true_e[test_idx], 0.10, 0.90)

            ipw_t_true = class_prior * Y_test[t_mask] / (
                e_star[t_mask] * (1.0 - e_star[t_mask])
            )
            ipw_u_true = -Y_test[u_mask] / (1.0 - e_star[u_mask])

            dm_u_true = (
                mu_t_hat[u_mask]
                - mu_u_hat[u_mask] / (1.0 - e_star[u_mask])
                + e_star[u_mask] * mu_t_hat[u_mask] / (1.0 - e_star[u_mask])
            )

            efficient_t_true = class_prior * (
                Y_test[t_mask] - mu_t_hat[t_mask]
            ) / (e_star[t_mask] * (1.0 - e_star[t_mask]))
            efficient_u_true = (
                -(Y_test[u_mask] - mu_u_hat[u_mask]) / (1.0 - e_star[u_mask])
                + dm_u_true
            )

            oracle["ipw_t_true_e"].append(ipw_t_true)
            oracle["ipw_u_true_e"].append(ipw_u_true)
            oracle["dm_u_true_e"].append(dm_u_true)
            oracle["efficient_t_true_e"].append(efficient_t_true)
            oracle["efficient_u_true_e"].append(efficient_u_true)

    ipw_t_all = np.concatenate(estimated["ipw_t"])
    ipw_u_all = np.concatenate(estimated["ipw_u"])
    dm_u_all = np.concatenate(estimated["dm_u"])
    eff_t_all = np.concatenate(estimated["efficient_t"])
    eff_u_all = np.concatenate(estimated["efficient_u"])

    estimates: OrderedDict[str, float] = OrderedDict()
    variances: OrderedDict[str, float] = OrderedDict()

    estimates["ipw"], variances["ipw"] = _case_control_group_stats(ipw_t_all, ipw_u_all)
    total_n = len(ipw_t_all) + len(ipw_u_all)
    alpha = len(ipw_t_all) / total_n
    estimates["dm"] = float(np.mean(dm_u_all))
    variances["dm"] = float(np.var(dm_u_all, ddof=0) / (1.0 - alpha))
    estimates["efficient"], variances["efficient"] = _case_control_group_stats(
        eff_t_all, eff_u_all
    )

    if oracle is not None:
        ipw_t_all = np.concatenate(oracle["ipw_t_true_e"])
        ipw_u_all = np.concatenate(oracle["ipw_u_true_e"])
        dm_u_all = np.concatenate(oracle["dm_u_true_e"])
        eff_t_all = np.concatenate(oracle["efficient_t_true_e"])
        eff_u_all = np.concatenate(oracle["efficient_u_true_e"])

        estimates["ipw_true_e"], variances["ipw_true_e"] = _case_control_group_stats(
            ipw_t_all, ipw_u_all
        )
        estimates["dm_true_e"] = float(np.mean(dm_u_all))
        variances["dm_true_e"] = float(np.var(dm_u_all, ddof=0) / (1.0 - alpha))
        estimates["efficient_true_e"], variances["efficient_true_e"] = _case_control_group_stats(
            eff_t_all, eff_u_all
        )

    return ATEEstimate(estimates=estimates, asymptotic_variances=variances)
