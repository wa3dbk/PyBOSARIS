"""
Optimization algorithms for PyBOSARIS.

This module provides optimization routines for calibration and fusion,
particularly for logistic regression.

Optimizers:
    Trust-region Newton Conjugate Gradient method
    L-BFGS
    Gradient descent with momentum

Functions:
    minimize_logistic: Optimize logistic regression objective
    trust_region_newton: Trust-region Newton-CG optimizer
"""

# from .trust_region import trust_region_newton
# from .objectives import logistic_regression_objective

__all__ = [
    # "trust_region_newton",
    # "logistic_regression_objective",
]
