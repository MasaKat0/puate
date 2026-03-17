
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from .data import (
    ExperimentData,
    generate_case_control_synthetic_linear,
    generate_case_control_synthetic_nonlinear,
    generate_censoring_synthetic_linear,
    generate_censoring_synthetic_nonlinear,
    make_case_control_ihdp_observations,
    make_censoring_ihdp_observations,
)
from .estimators import estimate_case_control_ate, estimate_censoring_ate
from .models import BoundedProbClassifier, LinearPULearner, NonNegativePULearner


@dataclass
class TrialResults:
    estimates: pd.DataFrame
    variances: pd.DataFrame
    true_ate: float
    ci_sample_size: int
    metadata: dict[str, Any]

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.estimates.to_csv(output_dir / "estimates.csv", index=False)
        self.variances.to_csv(output_dir / "variances.csv", index=False)
        payload = dict(self.metadata)
        payload.update({"true_ate": self.true_ate, "ci_sample_size": self.ci_sample_size})
        (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: str | Path) -> "TrialResults":
        output_dir = Path(output_dir)
        estimates = pd.read_csv(output_dir / "estimates.csv")
        variances = pd.read_csv(output_dir / "variances.csv")
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        true_ate = float(metadata.pop("true_ate"))
        ci_sample_size = int(metadata.pop("ci_sample_size"))
        return cls(
            estimates=estimates,
            variances=variances,
            true_ate=true_ate,
            ci_sample_size=ci_sample_size,
            metadata=metadata,
        )


def _has_cache(output_dir: str | Path) -> bool:
    output_dir = Path(output_dir)
    return all((output_dir / name).exists() for name in ["estimates.csv", "variances.csv", "metadata.json"])


def _censoring_linear_models():
    return (
        BoundedProbClassifier(LogisticRegression(max_iter=1000)),
        LinearRegression(),
        LinearRegression(),
    )


def _censoring_mlp_models():
    return (
        BoundedProbClassifier(MLPClassifier(random_state=1, max_iter=500)),
        MLPRegressor(random_state=1, max_iter=500),
        MLPRegressor(random_state=1, max_iter=500),
    )


def _case_control_linear_models(class_prior: float):
    return (
        BoundedProbClassifier(LinearPULearner(prior=class_prior)),
        LinearRegression(),
        LinearRegression(),
    )


def _case_control_mlp_models(class_prior: float):
    return (
        BoundedProbClassifier(NonNegativePULearner(prior=class_prior, random_state=1)),
        MLPRegressor(random_state=1, max_iter=500),
        MLPRegressor(random_state=1, max_iter=500),
    )


def _run_experiment(
    *,
    n_trials: int,
    make_data_fn,
    estimate_fn,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
    true_ate_override: float | None = None,
    ci_sample_size_override: int | None = None,
) -> TrialResults:
    if output_dir is not None and (not force) and _has_cache(output_dir):
        return TrialResults.load(output_dir)

    estimate_rows: list[dict[str, float]] = []
    variance_rows: list[dict[str, float]] = []
    true_ate: float | None = None
    ci_sample_size: int | None = None
    metadata: dict[str, Any] | None = None

    for trial in range(n_trials):
        if verbose and (trial == 0 or (trial + 1) % max(1, n_trials // 10) == 0):
            print(f"trial {trial + 1}/{n_trials}")

        rng = np.random.default_rng(base_seed + trial)
        data: ExperimentData = make_data_fn(rng)
        estimate = estimate_fn(data, base_seed + trial)

        estimate_rows.append(dict(estimate.estimates))
        variance_rows.append(dict(estimate.asymptotic_variances))

        if true_ate is None:
            true_ate = float(data.true_ate if true_ate_override is None else true_ate_override)
            ci_sample_size = int(
                data.metadata.get("sample_size", len(data.X))
                if ci_sample_size_override is None
                else ci_sample_size_override
            )
            metadata = dict(data.metadata)

    result = TrialResults(
        estimates=pd.DataFrame(estimate_rows),
        variances=pd.DataFrame(variance_rows),
        true_ate=float(true_ate),
        ci_sample_size=int(ci_sample_size),
        metadata=dict(metadata or {}),
    )

    if output_dir is not None:
        result.save(output_dir)

    return result


def _run_censoring_synthetic_linear(
    *,
    n_trials: int,
    n: int,
    p: int = 3,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    def make_data(rng):
        return generate_censoring_synthetic_linear(n=n, p=p, rng=rng)

    def estimate_one(data: ExperimentData, seed: int):
        obs_model, mu_t, mu_u = _censoring_linear_models()
        return estimate_censoring_ate(
            data.X,
            data.O,
            data.Y,
            obs_model,
            mu_t,
            mu_u,
            float(data.metadata["observation_rate"]),
            n_folds=2,
            random_state=seed,
            true_g=data.true_propensity,
        )

    return _run_experiment(
        n_trials=n_trials,
        make_data_fn=make_data,
        estimate_fn=estimate_one,
        output_dir=output_dir,
        force=force,
        base_seed=base_seed,
        verbose=verbose,
    )


def _run_censoring_synthetic_nonlinear(
    *,
    n_trials: int,
    n: int,
    p: int = 10,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    def make_data(rng):
        return generate_censoring_synthetic_nonlinear(n=n, p=p, rng=rng)

    def estimate_one(data: ExperimentData, seed: int):
        obs_model, mu_t, mu_u = _censoring_mlp_models()
        return estimate_censoring_ate(
            data.X,
            data.O,
            data.Y,
            obs_model,
            mu_t,
            mu_u,
            float(data.metadata["observation_rate"]),
            n_folds=2,
            random_state=seed,
            true_g=data.true_propensity,
        )

    return _run_experiment(
        n_trials=n_trials,
        make_data_fn=make_data,
        estimate_fn=estimate_one,
        output_dir=output_dir,
        force=force,
        base_seed=base_seed,
        verbose=verbose,
    )


def _run_case_control_synthetic_linear(
    *,
    n_trials: int,
    n_positive: int,
    n_unlabeled: int,
    p: int = 3,
    class_prior: float = 0.3,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    def make_data(rng):
        return generate_case_control_synthetic_linear(
            n_positive=n_positive,
            n_unlabeled=n_unlabeled,
            p=p,
            class_prior=class_prior,
            rng=rng,
        )

    def estimate_one(data: ExperimentData, seed: int):
        prop_model, mu_t, mu_u = _case_control_linear_models(class_prior)
        est = estimate_case_control_ate(
            data.X,
            data.O,
            data.Y,
            prop_model,
            mu_t,
            mu_u,
            class_prior=class_prior,
            n_folds=2,
            random_state=seed,
            true_e=data.true_propensity,
        )
        data.metadata["sample_size"] = int(n_positive + n_unlabeled)
        return est

    return _run_experiment(
        n_trials=n_trials,
        make_data_fn=make_data,
        estimate_fn=estimate_one,
        output_dir=output_dir,
        force=force,
        base_seed=base_seed,
        verbose=verbose,
    )


def _run_case_control_synthetic_nonlinear(
    *,
    n_trials: int,
    n_positive: int,
    n_unlabeled: int,
    p: int = 3,
    class_prior: float = 0.3,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    def make_data(rng):
        return generate_case_control_synthetic_nonlinear(
            n_positive=n_positive,
            n_unlabeled=n_unlabeled,
            p=p,
            class_prior=class_prior,
            rng=rng,
        )

    def estimate_one(data: ExperimentData, seed: int):
        prop_model, mu_t, mu_u = _case_control_mlp_models(class_prior)
        est = estimate_case_control_ate(
            data.X,
            data.O,
            data.Y,
            prop_model,
            mu_t,
            mu_u,
            class_prior=class_prior,
            n_folds=2,
            random_state=seed,
            true_e=data.true_propensity,
        )
        data.metadata["sample_size"] = int(n_positive + n_unlabeled)
        return est

    return _run_experiment(
        n_trials=n_trials,
        make_data_fn=make_data,
        estimate_fn=estimate_one,
        output_dir=output_dir,
        force=force,
        base_seed=base_seed,
        verbose=verbose,
    )


def _run_censoring_ihdp(
    *,
    surface: str,
    n_trials: int,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    def make_data(rng):
        return make_censoring_ihdp_observations(surface=surface, observation_rate=0.1, rng=rng)

    def estimate_one(data: ExperimentData, seed: int):
        obs_model, mu_t, mu_u = _censoring_mlp_models()
        return estimate_censoring_ate(
            data.X,
            data.O,
            data.Y,
            obs_model,
            mu_t,
            mu_u,
            0.1,
            n_folds=2,
            random_state=seed,
            true_g=None,
        )

    return _run_experiment(
        n_trials=n_trials,
        make_data_fn=make_data,
        estimate_fn=estimate_one,
        output_dir=output_dir,
        force=force,
        base_seed=base_seed,
        verbose=verbose,
        true_ate_override=4.0,
        ci_sample_size_override=747,
    )


def _run_case_control_ihdp(
    *,
    surface: str,
    n_trials: int,
    output_dir: str | Path | None = None,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    class_prior = 0.1

    def make_data(rng):
        return make_case_control_ihdp_observations(surface=surface, class_prior=class_prior, rng=rng)

    def estimate_one(data: ExperimentData, seed: int):
        prop_model, mu_t, mu_u = _case_control_mlp_models(class_prior)
        return estimate_case_control_ate(
            data.X,
            data.O,
            data.Y,
            prop_model,
            mu_t,
            mu_u,
            class_prior=class_prior,
            n_folds=2,
            random_state=seed,
            true_e=None,
        )

    return _run_experiment(
        n_trials=n_trials,
        make_data_fn=make_data,
        estimate_fn=estimate_one,
        output_dir=output_dir,
        force=force,
        base_seed=base_seed,
        verbose=verbose,
        true_ate_override=4.0,
        ci_sample_size_override=747,
    )



def run_synthetic_experiment(
    *,
    setting: str,
    model: str,
    n_trials: int,
    output_dir: str | Path,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
    n: int | None = None,
    p: int | None = None,
    n_positive: int | None = None,
    n_unlabeled: int | None = None,
    class_prior: float = 0.3,
) -> TrialResults:
    """Run a synthetic experiment and return saved trial results."""

    if setting == "censoring" and model == "linear":
        return _run_censoring_synthetic_linear(
            n_trials=n_trials,
            n=3000 if n is None else n,
            p=3 if p is None else p,
            output_dir=output_dir,
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        )
    if setting == "censoring" and model == "nonlinear":
        return _run_censoring_synthetic_nonlinear(
            n_trials=n_trials,
            n=3000 if n is None else n,
            p=10 if p is None else p,
            output_dir=output_dir,
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        )
    if setting == "case-control" and model == "linear":
        return _run_case_control_synthetic_linear(
            n_trials=n_trials,
            n_positive=1000 if n_positive is None else n_positive,
            n_unlabeled=2000 if n_unlabeled is None else n_unlabeled,
            p=3 if p is None else p,
            class_prior=class_prior,
            output_dir=output_dir,
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        )
    if setting == "case-control" and model == "nonlinear":
        return _run_case_control_synthetic_nonlinear(
            n_trials=n_trials,
            n_positive=1000 if n_positive is None else n_positive,
            n_unlabeled=2000 if n_unlabeled is None else n_unlabeled,
            p=3 if p is None else p,
            class_prior=class_prior,
            output_dir=output_dir,
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        )
    raise ValueError("Unsupported combination of setting and model.")


def run_ihdp_experiment(
    *,
    setting: str,
    surface: str,
    n_trials: int,
    output_dir: str | Path,
    force: bool = False,
    base_seed: int = 0,
    verbose: bool = True,
) -> TrialResults:
    """Run an IHDP experiment and return saved trial results."""

    if setting == "censoring":
        return _run_censoring_ihdp(
            surface=surface,
            n_trials=n_trials,
            output_dir=output_dir,
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        )
    if setting == "case-control":
        return _run_case_control_ihdp(
            surface=surface,
            n_trials=n_trials,
            output_dir=output_dir,
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        )
    raise ValueError("setting must be 'censoring' or 'case-control'")


def run_table_1_experiment(
    *,
    output_root: str | Path = "results/table_1",
    force: bool = False,
    n_trials: int = 5000,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict[str, TrialResults]:
    output_root = Path(output_root)
    return {
        "censoring": _run_censoring_synthetic_linear(
            n_trials=n_trials,
            n=3000,
            output_dir=output_root / "censoring",
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        ),
        "case_control": _run_case_control_synthetic_linear(
            n_trials=n_trials,
            n_positive=1000,
            n_unlabeled=2000,
            output_dir=output_root / "case_control",
            force=force,
            base_seed=base_seed + 100000,
            verbose=verbose,
        ),
    }


def run_table_2_experiment(
    *,
    output_root: str | Path = "results/table_2",
    force: bool = False,
    n_trials: int = 5000,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict[str, TrialResults]:
    output_root = Path(output_root)
    return {
        "censoring": _run_censoring_synthetic_nonlinear(
            n_trials=n_trials,
            n=3000,
            output_dir=output_root / "censoring",
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        ),
        "case_control": _run_case_control_synthetic_nonlinear(
            n_trials=n_trials,
            n_positive=1000,
            n_unlabeled=2000,
            output_dir=output_root / "case_control",
            force=force,
            base_seed=base_seed + 100000,
            verbose=verbose,
        ),
    }


def run_table_3_experiment(
    *,
    output_root: str | Path = "results/table_3",
    force: bool = False,
    n_trials: int = 5000,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict[str, TrialResults]:
    output_root = Path(output_root)
    return {
        "censoring": _run_censoring_synthetic_nonlinear(
            n_trials=n_trials,
            n=5000,
            output_dir=output_root / "censoring",
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        ),
        "case_control": _run_case_control_synthetic_nonlinear(
            n_trials=n_trials,
            n_positive=2000,
            n_unlabeled=3000,
            output_dir=output_root / "case_control",
            force=force,
            base_seed=base_seed + 100000,
            verbose=verbose,
        ),
    }


def run_table_4_experiment(
    *,
    output_root: str | Path = "results/table_4",
    force: bool = False,
    n_trials: int = 1000,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict[str, TrialResults]:
    output_root = Path(output_root)
    return {
        "censoring": _run_censoring_ihdp(
            surface="A",
            n_trials=n_trials,
            output_dir=output_root / "censoring",
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        ),
        "case_control": _run_case_control_ihdp(
            surface="A",
            n_trials=n_trials,
            output_dir=output_root / "case_control",
            force=force,
            base_seed=base_seed + 100000,
            verbose=verbose,
        ),
    }


def run_table_5_experiment(
    *,
    output_root: str | Path = "results/table_5",
    force: bool = False,
    n_trials: int = 1000,
    base_seed: int = 0,
    verbose: bool = True,
) -> dict[str, TrialResults]:
    output_root = Path(output_root)
    return {
        "censoring": _run_censoring_ihdp(
            surface="B",
            n_trials=n_trials,
            output_dir=output_root / "censoring",
            force=force,
            base_seed=base_seed,
            verbose=verbose,
        ),
        "case_control": _run_case_control_ihdp(
            surface="B",
            n_trials=n_trials,
            output_dir=output_root / "case_control",
            force=force,
            base_seed=base_seed + 100000,
            verbose=verbose,
        ),
    }
