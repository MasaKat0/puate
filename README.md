# PUATE

This is a replication code for ``PUATE: Efficient Average Treatment Effect Estimation from Treated (Positive) and Unlabeled Units
.''


```bibtex
@inproceedings{
kato2025puate,
title={{PUATE}: Efficient {ATE} Estimation from Treated (Positive) and Unlabeled Units},
author={Masahiro Kato and Fumiaki Kozai and Ryo Inokuchi},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Tl3Sg0SBEU}
}
```



## Repository layout

```text
.
├── src/puate/           # supported implementation
├── scripts/             # experiment entry points
├── tests/               # smoke tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Base install:

```bash
pip install -e .
```

IHDP experiments need `econml`:

```bash
pip install -e ".[ihdp]"
```

Or:

```bash
pip install -r requirements.txt
pip install econml
```

## Quick start

Synthetic censoring / linear:

```bash
python scripts/run_synthetic.py \
  --setting censoring \
  --model linear \
  --trials 10 \
  --output-dir results/generated/censoring_linear
```

Synthetic case-control / linear:

```bash
python scripts/run_synthetic.py \
  --setting case-control \
  --model linear \
  --trials 10 \
  --output-dir results/generated/case_control_linear
```

IHDP surface A / censoring:

```bash
python scripts/run_ihdp.py \
  --setting censoring \
  --surface A \
  --trials 10 \
  --output-dir results/generated/ihdp_censoring_A
```

Summarize a run:

```bash
python scripts/summarize_results.py \
  --run-dir results/generated/censoring_linear \
  --true-ate 3.0
```

## Supported modules

### `puate.estimators`

- `estimate_censoring_ate`
- `estimate_case_control_ate`

### `puate.models`

- `BoundedProbClassifier`
- `ElkanNotoClassifier`
- `LinearPULearner`
- `NonNegativePULearner`

### `puate.data`

- synthetic data generators for the censoring and case-control settings
- IHDP data wrappers for surfaces A and B

## Testing

```bash
pytest
```
