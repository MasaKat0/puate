#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated PUATE runs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing estimates.csv, variances.csv, and metadata.json.")
    parser.add_argument("--true-ate", type=float, required=True)
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size used in confidence intervals.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def infer_sample_size(metadata: dict[str, object]) -> int:
    if "sample_size" in metadata:
        return int(metadata["sample_size"])
    if "n_positive" in metadata and "n_unlabeled" in metadata:
        return int(metadata["n_positive"]) + int(metadata["n_unlabeled"])
    raise ValueError("Could not infer sample size from metadata. Pass --sample-size explicitly.")


def main() -> None:
    args = parse_args()
    estimates = pd.read_csv(args.run_dir / "estimates.csv")
    variances = pd.read_csv(args.run_dir / "variances.csv")

    with open(args.run_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n = args.sample_size or infer_sample_size(metadata)

    rows = []
    method_names = [c for c in estimates.columns if c != "trial"]
    for method in method_names:
        ate = estimates[method].to_numpy()
        var = variances[method].to_numpy()
        lower = ate - 1.96 * np.sqrt(var / n)
        upper = ate + 1.96 * np.sqrt(var / n)
        coverage = np.mean((lower <= args.true_ate) & (args.true_ate <= upper))
        rows.append(
            {
                "method": method,
                "mse": float(np.mean((ate - args.true_ate) ** 2)),
                "bias": float(np.mean(ate - args.true_ate)),
                "coverage_ratio": float(coverage),
            }
        )

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    if args.output is not None:
        summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
