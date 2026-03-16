#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from puate.data import (
    generate_case_control_synthetic_linear,
    generate_case_control_synthetic_nonlinear,
    generate_censoring_synthetic_linear,
    generate_censoring_synthetic_nonlinear,
)
from puate.estimators import estimate_case_control_ate, estimate_censoring_ate
from puate.models import BoundedProbClassifier, LinearPULearner, NonNegativePULearner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PUATE synthetic experiments.")
    parser.add_argument("--setting", choices=["censoring", "case-control"], required=True)
    parser.add_argument("--model", choices=["linear", "mlp"], required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None, help="Sample size for the censoring setting.")
    parser.add_argument("--n-positive", type=int, default=None, help="Positive-sample size for the case-control setting.")
    parser.add_argument("--n-unlabeled", type=int, default=None, help="Unlabeled-sample size for the case-control setting.")
    parser.add_argument("--p", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(args.seed)
    estimates_records: list[dict[str, float]] = []
    variance_records: list[dict[str, float]] = []
    metadata: dict[str, object] = {
        "setting": args.setting,
        "model": args.model,
        "trials": args.trials,
        "seed": args.seed,
        "n_folds": args.n_folds,
    }

    for trial in range(args.trials):
        trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        if args.setting == "censoring":
            if args.model == "linear":
                data = generate_censoring_synthetic_linear(
                    n=args.n or 3000,
                    p=args.p or 3,
                    rng=trial_rng,
                )
                obs_model = BoundedProbClassifier(LogisticRegression(max_iter=1000))
                mu_t_model = LinearRegression()
                mu_u_model = LinearRegression()
            else:
                data = generate_censoring_synthetic_nonlinear(
                    n=args.n or 3000,
                    p=args.p or 10,
                    rng=trial_rng,
                )
                obs_model = BoundedProbClassifier(MLPClassifier(random_state=1, max_iter=500))
                mu_t_model = MLPRegressor(random_state=1, max_iter=500)
                mu_u_model = MLPRegressor(random_state=1, max_iter=500)

            result = estimate_censoring_ate(
                data.X,
                data.O,
                data.Y,
                observation_model=obs_model,
                mu_t_model=mu_t_model,
                mu_u_model=mu_u_model,
                observation_rate=float(data.metadata["observation_rate"]),
                n_folds=args.n_folds,
                random_state=int(rng.integers(0, 2**32 - 1)),
                true_g=data.true_propensity,
            )

        else:
            if args.model == "linear":
                data = generate_case_control_synthetic_linear(
                    n_positive=args.n_positive or 1000,
                    n_unlabeled=args.n_unlabeled or 2000,
                    p=args.p or 3,
                    rng=trial_rng,
                )
                prop_model = BoundedProbClassifier(LinearPULearner(prior=float(data.metadata["class_prior"])))
                mu_t_model = LinearRegression()
                mu_u_model = LinearRegression()
            else:
                data = generate_case_control_synthetic_nonlinear(
                    n_positive=args.n_positive or 1000,
                    n_unlabeled=args.n_unlabeled or 2000,
                    p=args.p or 3,
                    rng=trial_rng,
                )
                prop_model = BoundedProbClassifier(NonNegativePULearner(prior=float(data.metadata["class_prior"])))
                mu_t_model = MLPRegressor(random_state=1, max_iter=500)
                mu_u_model = MLPRegressor(random_state=1, max_iter=500)

            result = estimate_case_control_ate(
                data.X,
                data.O,
                data.Y,
                propensity_model=prop_model,
                mu_t_model=mu_t_model,
                mu_u_model=mu_u_model,
                class_prior=float(data.metadata["class_prior"]),
                n_folds=args.n_folds,
                random_state=int(rng.integers(0, 2**32 - 1)),
                true_e=data.true_propensity,
            )

        estimates_records.append({"trial": trial, **result.estimates})
        variance_records.append({"trial": trial, **result.asymptotic_variances})
        metadata.update(data.metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(estimates_records).to_csv(args.output_dir / "estimates.csv", index=False)
    pd.DataFrame(variance_records).to_csv(args.output_dir / "variances.csv", index=False)
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
