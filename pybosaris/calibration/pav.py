"""
PAV (Pool Adjacent Violators) calibration for PyBOSARIS.

PAV is a non-parametric isotonic regression algorithm that fits a monotonically
non-decreasing function to data. It's used for score calibration to convert
scores to well-calibrated log-likelihood-ratios.
"""

import numpy as np
from typing import Tuple, Callable, Optional
import warnings

from ..core import Scores, Key
from ..utils import logit


def pavx(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool Adjacent Violators algorithm.

    Fits a vector with nondecreasing components to the data vector y
    such that sum((y - ghat)^2) is minimal.

    Parameters
    ----------
    y : ndarray
        Data vector to fit

    Returns
    -------
    ghat : ndarray
        Fitted monotonic vector (same length as y)
    width : ndarray
        Width of PAV bins, from left to right
    height : ndarray
        Corresponding heights of bins (in increasing order)

    Examples
    --------
    >>> y = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
    >>> ghat, width, height = pavx(y)
    >>> np.all(np.diff(ghat) >= 0)  # Check monotonicity
    True

    Notes
    -----
    This is a simplified version of the isotonic regression algorithm.
    The algorithm pools adjacent violators (pairs where the left value
    is greater than the right value) by replacing them with their mean.

    References
    ----------
    Based on code by Lutz Duembgen:
    http://www.imsv.unibe.ch/~duembgen/software
    """
    y = np.asarray(y).ravel()
    n = len(y)

    if n == 0:
        raise ValueError("Input array must not be empty")

    # Initialize
    index = np.zeros(n, dtype=int)
    length = np.zeros(n, dtype=int)
    ghat = np.zeros(n)

    # Start with first element
    ci = 0  # Current interval index (0-based in Python)
    index[ci] = 0
    length[ci] = 1
    ghat[ci] = y[0]

    # Process each element
    for j in range(1, n):
        # Create new interval for element j
        ci += 1
        index[ci] = j
        length[ci] = 1
        ghat[ci] = y[j]

        # Pool adjacent violators while necessary
        while ci >= 1 and ghat[ci - 1] >= ghat[ci]:
            # Merge intervals ci-1 and ci
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    # Extract final intervals
    num_intervals = ci + 1
    height = ghat[:num_intervals].copy()
    width = length[:num_intervals].copy()

    # Expand to full vector
    result = np.zeros(len(y))
    k = num_intervals - 1
    pos = len(y) - 1

    while k >= 0:
        start_idx = index[k]
        for j in range(start_idx, pos + 1):
            result[j] = ghat[k]
        pos = start_idx - 1
        k -= 1

    return result, width, height


def pav_calibration(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    score_offset: float = 1e-6
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
    """
    Create PAV calibration transformation from target and non-target scores.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)
    score_offset : float, optional
        Offset to make transformation monotonically increasing by tilting
        flat portions. Default: 1e-6

    Returns
    -------
    pav_transform : callable
        Transformation function that maps scores to calibrated LLRs
    score_bounds : ndarray
        Left and right ends of line segments making up the transformation
    llr_bounds : ndarray
        Lower and upper ends of line segments making up the transformation

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> pav_trans, score_bounds, llr_bounds = pav_calibration(tar, non)
    >>> # Apply to new scores
    >>> new_scores = np.array([0.0, 2.5, 5.0])
    >>> calibrated = pav_trans(new_scores)

    Notes
    -----
    The PAV algorithm creates a piecewise linear transformation that
    maps scores to log-likelihood-ratios. The transformation is
    monotonically increasing and optimally calibrated for the training data.

    The score_offset parameter adds a small tilt to flat portions of the
    transformation to ensure strict monotonicity, which is important for
    invertibility and numerical stability.
    """
    tar_scores = np.asarray(tar_scores).ravel()
    non_scores = np.asarray(non_scores).ravel()

    if len(tar_scores) == 0:
        raise ValueError("tar_scores must not be empty")
    if len(non_scores) == 0:
        raise ValueError("non_scores must not be empty")

    # Large value for extrapolation beyond observed scores
    large_val = 1e6

    # Combine scores with extreme values at ends
    scores = np.concatenate([
        [-large_val],
        tar_scores,
        non_scores,
        [large_val]
    ])

    # Ideal labels: 1 for targets (including left sentinel), 0 for non-targets
    ideal_labels = np.concatenate([
        np.ones(len(tar_scores) + 1),
        np.zeros(len(non_scores) + 1)
    ])

    # Sort by score
    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    ideal_labels = ideal_labels[sorted_indices]

    # Apply PAV to get optimal probabilities
    optimal_probs, width, height = pavx(ideal_labels)

    # Calculate data prior (fraction of targets in training data)
    data_prior = (len(tar_scores) + 1) / len(ideal_labels)

    # Convert probabilities to log-likelihood-ratios
    # LLR = log(P(target|score) / P(nontarget|score)) - log(P(target) / P(nontarget))
    # This removes the effect of the training data prior
    with np.errstate(divide='ignore', invalid='ignore'):
        llr = logit(optimal_probs) - logit(data_prior)

    # Handle infinities
    llr = np.clip(llr, -large_val, large_val)

    # Create boundary indices for piecewise linear function
    bnd_indices = _make_boundary_indices(width)

    # Extract score and LLR boundaries
    score_bounds = scores[bnd_indices]
    llr_bounds = llr[bnd_indices]

    # Tilt each flat portion by score_offset
    # Left ends: subtract offset, Right ends: add offset
    llr_bounds[::2] -= score_offset  # Even indices (0, 2, 4, ...) are left ends
    llr_bounds[1::2] += score_offset  # Odd indices (1, 3, 5, ...) are right ends

    # Create transformation function
    def pav_transform(s: np.ndarray) -> np.ndarray:
        """Apply PAV transformation to scores."""
        return _pav_transform_impl(s, score_bounds, llr_bounds)

    return pav_transform, score_bounds, llr_bounds


def _make_boundary_indices(width: np.ndarray) -> np.ndarray:
    """
    Create boundary indices from bin widths.

    For each bin, we need both the left and right boundaries.

    Parameters
    ----------
    width : ndarray
        Width of each bin

    Returns
    -------
    ndarray
        Boundary indices (interleaved left and right for each bin)
    """
    num_bins = len(width)
    bnd_indices = np.zeros(num_bins * 2, dtype=int)

    cumsum = np.cumsum(width)

    # Left boundaries: [0, cumsum[0], cumsum[1], ...]
    bnd_indices[::2] = np.concatenate([[0], cumsum[:-1]])

    # Right boundaries: [cumsum[0]-1, cumsum[1]-1, ...]
    bnd_indices[1::2] = cumsum - 1

    return bnd_indices


def _pav_transform_impl(
    scores: np.ndarray,
    score_bounds: np.ndarray,
    llr_bounds: np.ndarray
) -> np.ndarray:
    """
    Apply piecewise linear transformation.

    Parameters
    ----------
    scores : ndarray
        Scores to transform
    score_bounds : ndarray
        Score boundaries for piecewise linear function
    llr_bounds : ndarray
        LLR boundaries for piecewise linear function

    Returns
    -------
    ndarray
        Transformed scores (calibrated LLRs)
    """
    scores = np.asarray(scores)
    original_shape = scores.shape
    scores_flat = scores.ravel()

    result = np.zeros(len(scores_flat))

    for i, score in enumerate(scores_flat):
        # Find which segment this score falls in
        # We want the largest index where score >= score_bounds[index]
        idx = np.searchsorted(score_bounds, score, side='right') - 1

        # Clamp to valid range
        idx = max(0, min(idx, len(score_bounds) - 2))

        # Linear interpolation between boundaries
        x1 = score_bounds[idx]
        x2 = score_bounds[idx + 1]
        y1 = llr_bounds[idx]
        y2 = llr_bounds[idx + 1]

        if x2 == x1:
            # Avoid division by zero
            result[i] = y1
        else:
            # Linear interpolation
            result[i] = y1 + (y2 - y1) * (score - x1) / (x2 - x1)

    return result.reshape(original_shape)


def pav_calibrate_scores(
    scores: Scores,
    key: Key,
    prior: float = 0.5,
    score_offset: float = 1e-6
) -> Scores:
    """
    Calibrate scores using PAV algorithm.

    Trains a PAV transformation on the target and non-target scores
    and applies it to produce calibrated log-likelihood-ratios.

    Parameters
    ----------
    scores : Scores
        Scores object containing scores to calibrate
    key : Key
        Key object indicating target and non-target trials
    prior : float, optional
        Target prior for evaluation (default: 0.5)
        This is only used for reporting, not for calibration
    score_offset : float, optional
        Offset to make transformation strictly monotonic (default: 1e-6)

    Returns
    -------
    Scores
        Calibrated scores (as log-likelihood-ratios)

    Examples
    --------
    >>> # Create some scores
    >>> tar_mask = np.array([[True, False], [False, True]])
    >>> non_mask = np.array([[False, True], [True, False]])
    >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar_mask, non_mask)
    >>> score_mat = np.array([[2.0, -1.0], [-0.5, 3.0]])
    >>> scores = Scores(['m1', 'm2'], ['t1', 't2'], score_mat)
    >>> # Calibrate
    >>> calibrated = pav_calibrate_scores(scores, key)

    Notes
    -----
    This is "cheating" calibration - it trains and tests on the same data.
    For proper evaluation, use separate development and evaluation sets
    (see pav_calibrate_scores_dev_eval).

    The calibrated scores are log-likelihood-ratios where:
    - LLR > 0 indicates the trial is more likely to be a target
    - LLR < 0 indicates the trial is more likely to be a non-target
    - LLR = 0 means the score provides no information
    """
    # Validate inputs
    scores.validate()
    key.validate()

    if not 0 < prior < 1:
        raise ValueError(f"prior must be between 0 and 1, got {prior}")

    # Get target and non-target scores
    tar_scores, non_scores = scores.get_tar_non(key)

    if len(tar_scores) == 0:
        raise ValueError("No target scores found")
    if len(non_scores) == 0:
        raise ValueError("No non-target scores found")

    # Train PAV transformation
    pav_transform, score_bounds, llr_bounds = pav_calibration(
        tar_scores, non_scores, score_offset
    )

    # Align scores with key and apply transformation
    aligned_scores = scores.align_with_ndx(key)
    calibrated_scores = aligned_scores.transform(pav_transform)

    return calibrated_scores


def pav_calibrate_scores_dev_eval(
    dev_scores: Scores,
    dev_key: Key,
    eval_scores: Scores,
    eval_key: Key,
    prior: float = 0.5,
    score_offset: float = 1e-6
) -> Scores:
    """
    Calibrate scores using separate development and evaluation sets.

    Trains PAV transformation on development data and applies it to
    evaluation data. This is the proper way to evaluate calibration
    performance.

    Parameters
    ----------
    dev_scores : Scores
        Development scores for training calibration
    dev_key : Key
        Development key with target/non-target labels
    eval_scores : Scores
        Evaluation scores to calibrate
    eval_key : Key
        Evaluation key (for alignment only)
    prior : float, optional
        Target prior for evaluation (default: 0.5)
    score_offset : float, optional
        Offset for strict monotonicity (default: 1e-6)

    Returns
    -------
    Scores
        Calibrated evaluation scores

    Examples
    --------
    >>> # Development data
    >>> dev_tar_mask = np.array([[True, False]])
    >>> dev_non_mask = np.array([[False, True]])
    >>> dev_key = Key(['m1'], ['t1', 't2'], dev_tar_mask, dev_non_mask)
    >>> dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
    >>> # Evaluation data
    >>> eval_tar_mask = np.array([[True, False]])
    >>> eval_non_mask = np.array([[False, True]])
    >>> eval_key = Key(['m2'], ['t1', 't2'], eval_tar_mask, eval_non_mask)
    >>> eval_scores = Scores(['m2'], ['t1', 't2'], np.array([[1.5, -0.5]]))
    >>> # Calibrate
    >>> calibrated = pav_calibrate_scores_dev_eval(
    ...     dev_scores, dev_key, eval_scores, eval_key
    ... )

    Notes
    -----
    This function provides unbiased calibration evaluation by keeping
    training (development) and testing (evaluation) data separate.
    """
    # Validate inputs
    dev_scores.validate()
    dev_key.validate()
    eval_scores.validate()
    eval_key.validate()

    if not 0 < prior < 1:
        raise ValueError(f"prior must be between 0 and 1, got {prior}")

    # Get development target and non-target scores
    tar_scores, non_scores = dev_scores.get_tar_non(dev_key)

    if len(tar_scores) == 0:
        raise ValueError("No target scores in development data")
    if len(non_scores) == 0:
        raise ValueError("No non-target scores in development data")

    # Train PAV transformation on development data
    pav_transform, score_bounds, llr_bounds = pav_calibration(
        tar_scores, non_scores, score_offset
    )

    # Apply transformation to evaluation scores
    aligned_eval = eval_scores.align_with_ndx(eval_key)
    calibrated_eval = aligned_eval.transform(pav_transform)

    return calibrated_eval
