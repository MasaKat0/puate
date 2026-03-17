# PUATE

Replication code for *PUATE: Efficient ATE Estimation from Treated (Positive) and Unlabeled Units*.  
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
├── notebooks/           # one notebook per paper table / figure
├── results/             # generated outputs
├── scripts/             # command-line entry points
├── src/puate/           # implementation
├── tests/
├── pyproject.toml
├── requirements.txt
├── requirements-ihdp.txt
└── README.md
```

## Installation

Base install:

```bash
pip install -e .
```

IHDP experiments require `econml`:

```bash
pip install -e ".[ihdp]"
```

Or:

```bash
pip install -r requirements-ihdp.txt
```

## Notebook interface

The main interface of this repository is the `notebooks/` directory.  
Each paper table and figure has its own notebook.

- `notebooks/Table 1.ipynb`: synthetic linear experiment
- `notebooks/Figure 2.ipynb`: density plot for the synthetic linear experiment
- `notebooks/Table 2.ipynb`: synthetic nonlinear experiment, first sample-size setting
- `notebooks/Figure 3.ipynb`: density plot for the first nonlinear setting
- `notebooks/Table 3.ipynb`: synthetic nonlinear experiment, second sample-size setting
- `notebooks/Figure 4.ipynb`: density plot for the second nonlinear setting
- `notebooks/Table 4.ipynb`: IHDP response surface A
- `notebooks/Figure 5.ipynb`: density plot for IHDP response surface A
- `notebooks/Table 5.ipynb`: IHDP response surface B
- `notebooks/Figure 6.ipynb`: density plot for IHDP response surface B

To run a notebook:

```bash
jupyter lab
```

Then open the notebook you want and run all cells.  
Each notebook writes experiment outputs under `results/`.

## Command-line entry points

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

## Package overview

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
- IHDP data wrappers for response surfaces A and B

### `puate.paper_reproduction`

- `run_synthetic_experiment`
- `run_ihdp_experiment`
- `run_table_1_experiment`
- `run_table_2_experiment`
- `run_table_3_experiment`
- `run_table_4_experiment`
- `run_table_5_experiment`

## Testing

```bash
pytest
```

## Notes

- Full notebook runs use the default trial counts in each notebook and can take time.
- IHDP notebooks and IHDP command-line runs require `econml`.
