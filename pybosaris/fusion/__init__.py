"""
Multi-system fusion for PyBOSARIS.

This module provides algorithms for fusing scores from multiple subsystems
into a single calibrated log-likelihood-ratio output.

Methods:
    Linear fusion: Weighted combination of system scores

Functions:
    train_linear_fusion: Train fusion weights from score matrices
    linear_fuse_scores: Fuse multiple systems (same-data)
    linear_fuse_scores_dev_eval: Fuse with separate dev/eval sets
    stack_scores: Stack scores from multiple systems into a matrix
"""

from .linear_fusion import (
    train_linear_fusion,
    linear_fuse_scores,
    linear_fuse_scores_dev_eval,
    stack_scores
)

__all__ = [
    "train_linear_fusion",
    "linear_fuse_scores",
    "linear_fuse_scores_dev_eval",
    "stack_scores",
]
