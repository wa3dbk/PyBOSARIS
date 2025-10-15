"""
Core data structures for PyBOSARIS.

This module contains the fundamental classes for representing:
- Trial lists (Ndx, Key)
- Scores
- Quality measures
- Evaluation results

Classes:
    Ndx: Index of trials (model-test segment pairs)
    Key: Supervised trial list with target/non-target labels
    Scores: Score storage and manipulation
    Quality: Quality measures for trials
    SegQuality: Quality measures for segments
    Results: Container for evaluation results
"""

from .ndx import Ndx
from .key import Key
from .scores import Scores
# from .quality import Quality, SegQuality
# from .results import Results

__all__ = [
    "Ndx",
    "Key",
    "Scores",
    # "Quality",
    # "SegQuality",
    # "Results",
]
