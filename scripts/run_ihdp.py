#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from puate.paper_reproduction import run_ihdp_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IHDP PUATE experiments.")
    parser.add_argument("--setting", choices=["censoring", "case-control"], required=True)
    parser.add_argument("--surface", choices=["A", "B"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = run_ihdp_experiment(
        setting=args.setting,
        surface=args.surface,
        n_trials=args.trials,
        output_dir=args.output_dir,
        force=args.force,
        base_seed=args.seed,
        verbose=True,
    )

    print(f"Saved estimates to: {args.output_dir}")
    print(f"Methods: {', '.join(result.estimates.columns)}")


if __name__ == "__main__":
    main()
