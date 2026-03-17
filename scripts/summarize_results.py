#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from puate.paper_reproduction import TrialResults
from puate.reporting import summarize_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a saved PUATE run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--true-ate", type=float, default=None, help="Override the true ATE stored in metadata.")
    parser.add_argument("--ci-sample-size", type=int, default=None, help="Override the CI sample size stored in metadata.")
    args = parser.parse_args()

    result = TrialResults.load(args.run_dir)
    true_ate = result.true_ate if args.true_ate is None else args.true_ate
    ci_sample_size = result.ci_sample_size if args.ci_sample_size is None else args.ci_sample_size

    summary = summarize_trials(
        result.estimates,
        result.variances,
        true_ate=true_ate,
        ci_sample_size=ci_sample_size,
    )
    print(summary.round(4).to_string())
    print()
    print(json.dumps({"true_ate": true_ate, "ci_sample_size": ci_sample_size, **result.metadata}, indent=2))


if __name__ == "__main__":
    main()
