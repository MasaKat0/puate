#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from puate.data import make_case_control_ihdp_observations, make_censoring_ihdp_observations
from puate.estimators import estimate_case_control_ate, estimate_censoring_ate
from puate.models import BoundedProbClassifier, NonNegativePULearner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IHDP PUATE experiments.")
    parser.add_argument("--setting", choices=["censoring", "case-control"], required=True)
    parser.add_argument("--surface", choices=["A", "B"], required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(args.seed)
    estimates_records: list[dict[str, float]] = []
    variance_records: list[dict[str, float]] = []
    metadata: dict[str, object] = {
        "setting": args.setting,
        "surface": args.surface,
        "trials": args.trials,
        "seed": args.seed,
        "n_folds": args.n_folds,
    }

    for trial in range(args.trials):
        trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        if args.setting == "censoring":
            data = make_censoring_ihdp_observations(surface=args.surface, rng=trial_rng)
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
            )
        else:
            data = make_case_control_ihdp_observations(surface=args.surface, rng=trial_rng)
            prop_model = BoundedProbClassifier(
                NonNegativePULearner(prior=float(data.metadata["class_prior"]))
            )
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
            )

        estimates_records.append({"trial": trial, **result.estimates})
        variance_records.append({"trial": trial, **result.asymptotic_variances})
        metadata.update(data.metadata)
        metadata["true_ate_last_trial"] = data.true_ate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(estimates_records).to_csv(args.output_dir / "estimates.csv", index=False)
    pd.DataFrame(variance_records).to_csv(args.output_dir / "variances.csv", index=False)
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
