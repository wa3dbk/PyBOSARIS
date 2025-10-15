"""
Visualization and plotting functions for PyBOSARIS.

This module provides functions for creating publication-quality plots
of classifier performance.

Plot Types:
    DET Plot: Detection Error Tradeoff curve

Functions:
    plot_det_curve: Create single DET curve
    plot_det_curves: Create multiple DET curves
    plot_det_from_scores: Create DET curve from Scores/Key objects
    save_det_plot: Save current plot to file

Note: Requires matplotlib. Install with: pip install matplotlib
"""

from .det_curve import (
    plot_det_curve,
    plot_det_curves,
    plot_det_from_scores,
    save_det_plot,
    HAS_MATPLOTLIB
)

__all__ = [
    "plot_det_curve",
    "plot_det_curves",
    "plot_det_from_scores",
    "save_det_plot",
    "HAS_MATPLOTLIB",
]
