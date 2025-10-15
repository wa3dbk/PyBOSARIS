"""
Score calibration methods for PyBOSARIS.

This module provides algorithms for mapping uncalibrated scores to
well-calibrated log-likelihood-ratios.

Methods:
    PAV (Pool Adjacent Violators): Non-parametric isotonic regression
    Logistic Regression: Parametric affine transformation

Functions:
    PAV Calibration:
        pav_calibrate_scores: Apply PAV calibration (same-data training/testing)
        pav_calibrate_scores_dev_eval: Apply PAV calibration (separate dev/eval)
        pav_calibration: Train PAV transformation from scores
        pavx: Core Pool Adjacent Violators algorithm

    Logistic Regression Calibration:
        logistic_calibrate_scores: Apply logistic calibration (same-data)
        logistic_calibrate_scores_dev_eval: Apply logistic calibration (dev/eval)
        logistic_calibration: Train logistic transformation from scores
"""

from .pav import (
    pavx,
    pav_calibration,
    pav_calibrate_scores,
    pav_calibrate_scores_dev_eval
)
from .logistic import (
    logistic_calibration,
    logistic_calibrate_scores,
    logistic_calibrate_scores_dev_eval
)

__all__ = [
    # PAV calibration
    "pavx",
    "pav_calibration",
    "pav_calibrate_scores",
    "pav_calibrate_scores_dev_eval",
    # Logistic regression calibration
    "logistic_calibration",
    "logistic_calibrate_scores",
    "logistic_calibrate_scores_dev_eval",
]
