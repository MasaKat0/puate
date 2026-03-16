"""PUATE: Efficient ATE estimation from treated (positive) and unlabeled units.

The supported public API lives in :mod:`puate.estimators`, :mod:`puate.models`,
and :mod:`puate.data`.
"""

from .estimators import (
    ATEEstimate,
    estimate_case_control_ate,
    estimate_censoring_ate,
)
from .models import (
    BoundedProbClassifier,
    ElkanNotoClassifier,
    LinearPULearner,
    NonNegativePULearner,
)
from .data import (
    generate_case_control_synthetic_linear,
    generate_case_control_synthetic_nonlinear,
    generate_censoring_synthetic_linear,
    generate_censoring_synthetic_nonlinear,
    make_case_control_ihdp_observations,
    make_censoring_ihdp_observations,
)

__all__ = [
    "ATEEstimate",
    "estimate_case_control_ate",
    "estimate_censoring_ate",
    "BoundedProbClassifier",
    "ElkanNotoClassifier",
    "LinearPULearner",
    "NonNegativePULearner",
    "generate_case_control_synthetic_linear",
    "generate_case_control_synthetic_nonlinear",
    "generate_censoring_synthetic_linear",
    "generate_censoring_synthetic_nonlinear",
    "make_case_control_ihdp_observations",
    "make_censoring_ihdp_observations",
]
