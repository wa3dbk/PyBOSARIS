"""
Logistic regression (linear) calibration for PyBOSARIS.

Logistic regression calibration learns an affine transformation (scaling and
offset) that maps scores to calibrated log-likelihood-ratios by minimizing
the calibration loss (Cllr).
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy.optimize import minimize
import warnings

from ..core import Scores, Key
from ..utils import sigmoid, neg_log_sigmoid


def logistic_calibration(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    prior: float = 0.5,
    max_iter: int = 100
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Train logistic regression calibration from target and non-target scores.

    Learns an affine transformation (alpha * score + beta) that minimizes
    the calibration loss (log-likelihood loss / cross-entropy).

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)
    prior : float, optional
        Target prior for calibration (default: 0.5)
    max_iter : int, optional
        Maximum number of optimization iterations (default: 100)

    Returns
    -------
    calibration_func : callable
        Function that maps scores to calibrated LLRs
    weights : ndarray
        Calibration weights [alpha, beta] where
        calibrated_llr = alpha * score + beta

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> cal_func, weights = logistic_calibration(tar, non)
    >>> # Apply to new scores
    >>> new_scores = np.array([0.0, 2.5, 5.0])
    >>> calibrated = cal_func(new_scores)
    >>> print(f"Weights (alpha, beta): {weights}")

    Notes
    -----
    The calibration minimizes the log-likelihood loss (also called
    calibration loss or Cllr). The resulting transformation is:

        LLR = alpha * score + beta

    where alpha is the scaling factor and beta is the offset.

    This is equivalent to logistic regression with one feature (the score).
    The optimization uses L-BFGS-B algorithm for numerical stability.
    """
    tar_scores = np.asarray(tar_scores).ravel()
    non_scores = np.asarray(non_scores).ravel()

    if len(tar_scores) == 0:
        raise ValueError("tar_scores must not be empty")
    if len(non_scores) == 0:
        raise ValueError("non_scores must not be empty")
    if not 0 < prior < 1:
        raise ValueError(f"prior must be between 0 and 1, got {prior}")

    # Combine scores and create labels
    all_scores = np.concatenate([tar_scores, non_scores])
    labels = np.concatenate([
        np.ones(len(tar_scores)),    # 1 for targets
        np.zeros(len(non_scores))    # 0 for non-targets
    ])

    # Define objective function: negative log-likelihood (cross-entropy)
    def objective(w):
        """
        Negative log-likelihood loss.

        w[0] = alpha (scaling)
        w[1] = beta (offset)
        """
        alpha, beta = w

        # Compute calibrated scores (log-odds after calibration)
        calibrated = alpha * all_scores + beta

        # Compute log-likelihood loss
        # For target: -log(sigmoid(calibrated))
        # For non-target: -log(1 - sigmoid(calibrated)) = -log(sigmoid(-calibrated))

        # Use numerically stable version
        loss_tar = np.sum(neg_log_sigmoid(calibrated[:len(tar_scores)]))
        loss_non = np.sum(neg_log_sigmoid(-calibrated[len(tar_scores):]))

        # Total loss (we want to minimize this)
        total_loss = loss_tar + loss_non

        # Normalize by number of trials
        return total_loss / len(all_scores)

    # Initialize weights
    # Start with alpha=1, beta=logit(prior) for reasonable starting point
    # This gives us LLR ≈ score + log(prior / (1-prior))
    from ..utils import logit
    w0 = np.array([1.0, logit(prior)])

    # Optimize using L-BFGS-B
    result = minimize(
        objective,
        w0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )

    if not result.success:
        warnings.warn(
            f"Logistic calibration optimization did not converge: {result.message}"
        )

    weights = result.x
    alpha, beta = weights

    # Create calibration function
    def calibration_func(scores: np.ndarray) -> np.ndarray:
        """Apply logistic calibration to scores."""
        scores = np.asarray(scores)
        return alpha * scores + beta

    return calibration_func, weights


def logistic_calibrate_scores(
    scores: Scores,
    key: Key,
    prior: float = 0.5,
    max_iter: int = 100
) -> Tuple[Scores, np.ndarray]:
    """
    Calibrate scores using logistic regression.

    Trains a linear calibration (scaling and offset) on the target and
    non-target scores and applies it to produce calibrated LLRs.

    Parameters
    ----------
    scores : Scores
        Scores object containing scores to calibrate
    key : Key
        Key object indicating target and non-target trials
    prior : float, optional
        Target prior for calibration (default: 0.5)
    max_iter : int, optional
        Maximum optimization iterations (default: 100)

    Returns
    -------
    calibrated_scores : Scores
        Calibrated scores (as log-likelihood-ratios)
    weights : ndarray
        Calibration weights [alpha, beta]

    Examples
    --------
    >>> # Create some scores
    >>> tar_mask = np.array([[True, False], [False, True]])
    >>> non_mask = np.array([[False, True], [True, False]])
    >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar_mask, non_mask)
    >>> score_mat = np.array([[2.0, -1.0], [-0.5, 3.0]])
    >>> scores = Scores(['m1', 'm2'], ['t1', 't2'], score_mat)
    >>> # Calibrate
    >>> calibrated, weights = logistic_calibrate_scores(scores, key)
    >>> print(f"Alpha (scaling): {weights[0]:.3f}")
    >>> print(f"Beta (offset): {weights[1]:.3f}")

    Notes
    -----
    This is "cheating" calibration - it trains and tests on the same data.
    For proper evaluation, use separate development and evaluation sets
    (see logistic_calibrate_scores_dev_eval).

    The calibrated scores are log-likelihood-ratios of the form:
        LLR = alpha * score + beta

    where alpha and beta are learned to minimize the calibration loss.
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

    # Train logistic calibration
    calibration_func, weights = logistic_calibration(
        tar_scores, non_scores, prior, max_iter
    )

    # Align scores with key and apply calibration
    aligned_scores = scores.align_with_ndx(key)
    calibrated_scores = aligned_scores.transform(calibration_func)

    return calibrated_scores, weights


def logistic_calibrate_scores_dev_eval(
    dev_scores: Scores,
    dev_key: Key,
    eval_scores: Scores,
    eval_key: Key,
    prior: float = 0.5,
    max_iter: int = 100
) -> Tuple[Scores, np.ndarray]:
    """
    Calibrate scores using separate development and evaluation sets.

    Trains logistic calibration on development data and applies it to
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
        Target prior for calibration (default: 0.5)
    max_iter : int, optional
        Maximum optimization iterations (default: 100)

    Returns
    -------
    calibrated_scores : Scores
        Calibrated evaluation scores
    weights : ndarray
        Calibration weights [alpha, beta]

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
    >>> calibrated, weights = logistic_calibrate_scores_dev_eval(
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

    # Train logistic calibration on development data
    calibration_func, weights = logistic_calibration(
        tar_scores, non_scores, prior, max_iter
    )

    # Apply calibration to evaluation scores
    aligned_eval = eval_scores.align_with_ndx(eval_key)
    calibrated_eval = aligned_eval.transform(calibration_func)

    return calibrated_eval, weights
