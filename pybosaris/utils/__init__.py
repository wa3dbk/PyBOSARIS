"""
Utility functions for PyBOSARIS.

This module provides helper functions used across the toolkit.

Categories:
    Mathematical utilities (logit, sigmoid, effective_prior)
    Data manipulation (sorting, filtering)
    Validation functions
    Logging utilities

Functions:
    logit: Compute log-odds
    sigmoid: Compute inverse logit
    effective_prior: Calculate effective prior from costs
    probit: Compute probit transformation
    inv_probit: Compute inverse probit
"""

from .math_utils import (
    logit,
    sigmoid,
    probit,
    inv_probit,
    effective_prior,
    neg_log_sigmoid,
    safe_log,
)
from .validation import (
    validate_scores,
    validate_labels,
    validate_trial_mask,
    validate_key_consistency,
    validate_probability,
    validate_positive,
    validate_model_test_names,
    check_sufficient_trials,
)

__all__ = [
    # Math utilities
    "logit",
    "sigmoid",
    "probit",
    "inv_probit",
    "effective_prior",
    "neg_log_sigmoid",
    "safe_log",
    # Validation functions
    "validate_scores",
    "validate_labels",
    "validate_trial_mask",
    "validate_key_consistency",
    "validate_probability",
    "validate_positive",
    "validate_model_test_names",
    "check_sufficient_trials",
]
