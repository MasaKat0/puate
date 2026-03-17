#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from puate.paper_reproduction import run_synthetic_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic PUATE experiments.")
    parser.add_argument("--setting", choices=["censoring", "case-control"], required=True)
    parser.add_argument("--model", choices=["linear", "nonlinear"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n", type=int, default=None, help="Sample size for the censoring setting.")
    parser.add_argument("--p", type=int, default=None, help="Number of covariates.")
    parser.add_argument("--n-positive", type=int, default=None, help="Number of labeled positive samples.")
    parser.add_argument("--n-unlabeled", type=int, default=None, help="Number of unlabeled samples.")
    parser.add_argument("--class-prior", type=float, default=0.3, help="Positive class prior in the unlabeled sample.")
    args = parser.parse_args()

    result = run_synthetic_experiment(
        setting=args.setting,
        model=args.model,
        n_trials=args.trials,
        output_dir=args.output_dir,
        force=args.force,
        base_seed=args.seed,
        verbose=True,
        n=args.n,
        p=args.p,
        n_positive=args.n_positive,
        n_unlabeled=args.n_unlabeled,
        class_prior=args.class_prior,
    )

    print(f"Saved estimates to: {args.output_dir}")
    print(f"Methods: {', '.join(result.estimates.columns)}")


if __name__ == "__main__":
    main()
