"""
Performance evaluation metrics for PyBOSARIS.

This module provides metrics for assessing the quality of binary classification
scores, with emphasis on speaker recognition and biometric systems.

Metrics:
    EER: Equal Error Rate
    minDCF: Minimum Detection Cost Function
    Cllr: Calibration loss / log-likelihood ratio cost
    minCllr: Minimum calibration loss
    ROCCH: ROC Convex Hull

Functions:
    compute_eer: Calculate Equal Error Rate
    compute_min_dcf: Calculate minimum Detection Cost Function
    compute_cllr: Calculate calibration loss (for LLR scores)
    compute_min_cllr: Calculate minimum calibration loss
    compute_all_metrics: Calculate all metrics at once
    rocch: Compute ROC Convex Hull
    rocch2eer: Convert ROCCH to EER
"""

from .metrics import (
    rocch,
    rocch2eer,
    compute_eer,
    compute_min_dcf,
    compute_act_dcf,
    compute_cllr,
    compute_min_cllr,
    compute_all_metrics
)

__all__ = [
    "rocch",
    "rocch2eer",
    "compute_eer",
    "compute_min_dcf",
    "compute_act_dcf",
    "compute_cllr",
    "compute_min_cllr",
    "compute_all_metrics",
]
