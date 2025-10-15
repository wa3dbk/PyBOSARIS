"""
PyBOSARIS: Python implementation of the BOSARIS Toolkit

A comprehensive toolkit for calibrating, fusing and evaluating scores from
binary classifiers, with a focus on speaker recognition and biometric systems.

The toolkit provides:
- Score calibration (PAV, logistic regression)
- Multi-system fusion
- Performance evaluation (EER, minDCF, Cllr, ROCCH)
- Visualization (DET plots, normalized Bayes error-rate plots)
- Efficient binary file format (HDF5)

Based on the original MATLAB BOSARIS Toolkit by Niko Brümmer and Edward de Villiers.
Reference: "The BOSARIS Toolkit User Guide: Theory, Algorithms and Code for Binary
Classifier Score Processing" (December 2011)
"""

__version__ = "1.0.0"
__author__ = "PyBOSARIS Contributors"
__license__ = "MIT"

# Import core classes
from .core import Ndx, Key, Scores

# Import calibration functions
from .calibration import pav_calibrate_scores, logistic_calibrate_scores

# Import fusion functions
from .fusion import linear_fuse_scores

# Import evaluation metrics
from .evaluation import (
    compute_eer,
    compute_min_dcf,
    compute_act_dcf,
    compute_cllr,
    compute_min_cllr,
    compute_all_metrics,
    rocch
)

# Plotting functions are optional (requires matplotlib)
try:
    from .plotting import plot_det_curve, plot_det_curves, plot_det_from_scores
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core classes
    "Ndx",
    "Key",
    "Scores",
    # Calibration
    "pav_calibrate_scores",
    "logistic_calibrate_scores",
    # Fusion
    "linear_fuse_scores",
    # Evaluation
    "compute_eer",
    "compute_min_dcf",
    "compute_act_dcf",
    "compute_cllr",
    "compute_min_cllr",
    "compute_all_metrics",
    "rocch",
    # Plotting (if available)
    "HAS_PLOTTING",
]

# Add plotting functions to __all__ if matplotlib is available
if HAS_PLOTTING:
    __all__.extend([
        "plot_det_curve",
        "plot_det_curves",
        "plot_det_from_scores",
    ])
