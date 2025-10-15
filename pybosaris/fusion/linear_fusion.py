"""
Linear score fusion for PyBOSARIS.

Linear fusion combines scores from multiple systems using a weighted sum
to produce fused scores that are better calibrated and more accurate than
individual system scores.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from scipy.optimize import minimize
import warnings

from ..core import Scores, Key
from ..utils import sigmoid, neg_log_sigmoid


def stack_scores(
    scores_list: List[Scores],
    key: Key
) -> np.ndarray:
    """
    Stack scores from multiple systems into a matrix.

    Parameters
    ----------
    scores_list : list of Scores
        List of Scores objects, one per system
    key : Key
        Key object indicating target and non-target trials

    Returns
    -------
    ndarray
        Array of shape (n_systems, n_trials) with aligned scores

    Examples
    --------
    >>> scores1 = Scores(['m1'], ['t1', 't2'], np.array([[1.0, 2.0]]))
    >>> scores2 = Scores(['m1'], ['t1', 't2'], np.array([[3.0, 4.0]]))
    >>> tar_mask = np.array([[True, False]])
    >>> non_mask = np.array([[False, True]])
    >>> key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)
    >>> stacked = stack_scores([scores1, scores2], key)
    >>> stacked.shape
    (2, 2)

    Notes
    -----
    All systems must have scores for all trials in the key.
    Scores are aligned with the key before stacking.
    """
    # Get total number of trials
    trial_mask = key.tar_mask | key.non_mask
    n_trials = int(np.sum(trial_mask))
    n_systems = len(scores_list)

    # Create stacked matrix
    stacked = np.zeros((n_systems, n_trials))

    for i, scores in enumerate(scores_list):
        # Align scores with key
        aligned = scores.align_with_ndx(key)

        # Check that all trials have scores
        if aligned.n_scores != n_trials:
            raise ValueError(
                f"System {i} has {aligned.n_scores} scores, "
                f"but key has {n_trials} trials"
            )

        # Extract scores for trials
        scores_flat = aligned.score_mat.ravel()
        trial_mask_flat = trial_mask.ravel()
        stacked[i, :] = scores_flat[trial_mask_flat]

    return stacked


def train_linear_fusion(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    prior: float = 0.5,
    max_iter: int = 100
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Train linear fusion weights from target and non-target score matrices.

    The fusion computes:
        fused_score = w[0] * system_1 + w[1] * system_2 + ... + w[M-1] * system_M + w[M]

    where w[M] is the offset.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores, shape (n_systems, n_targets)
    non_scores : ndarray
        Non-target trial scores, shape (n_systems, n_nontargets)
    prior : float, optional
        Target prior for optimization (default: 0.5)
    max_iter : int, optional
        Maximum optimization iterations (default: 100)

    Returns
    -------
    fusion_func : callable
        Function that takes score matrix (n_systems, n_trials) and returns
        fused scores (n_trials,)
    weights : ndarray
        Fusion weights of shape (n_systems + 1,) where last element is offset

    Examples
    --------
    >>> # Two systems
    >>> tar1 = np.array([[2.0, 3.0], [1.5, 2.5]])  # 2 systems, 2 targets
    >>> non1 = np.array([[-1.0, -2.0], [-0.5, -1.5]])  # 2 systems, 2 non-targets
    >>> fusion_func, weights = train_linear_fusion(tar1, non1)
    >>> # Apply to new data
    >>> test_scores = np.array([[1.0, 2.0], [0.5, 1.5]])  # 2 systems, 2 trials
    >>> fused = fusion_func(test_scores)

    Notes
    -----
    The fusion is trained to minimize log-likelihood loss (Cllr).
    This is similar to logistic regression but with multiple input features.
    """
    tar_scores = np.asarray(tar_scores)
    non_scores = np.asarray(non_scores)

    if tar_scores.ndim != 2:
        raise ValueError("tar_scores must be 2D (n_systems × n_targets)")
    if non_scores.ndim != 2:
        raise ValueError("non_scores must be 2D (n_systems × n_nontargets)")

    n_systems_tar = tar_scores.shape[0]
    n_systems_non = non_scores.shape[0]

    if n_systems_tar != n_systems_non:
        raise ValueError(
            f"Number of systems must match: {n_systems_tar} vs {n_systems_non}"
        )

    n_systems = n_systems_tar

    if not 0 < prior < 1:
        raise ValueError(f"prior must be in (0, 1), got {prior}")

    # Combine scores
    all_scores = np.hstack([tar_scores, non_scores])  # (n_systems, n_total)
    n_tar = tar_scores.shape[1]
    n_non = non_scores.shape[1]

    # Labels: 1 for target, 0 for non-target
    labels = np.concatenate([
        np.ones(n_tar),
        np.zeros(n_non)
    ])

    # Define objective function: negative log-likelihood
    def objective(w):
        """
        Negative log-likelihood loss for fusion.

        w[0:n_systems] = weights for each system
        w[n_systems] = offset
        """
        # Compute fused scores: w^T @ scores + offset
        fused = w[:-1] @ all_scores + w[-1]

        # Compute log-likelihood loss
        loss_tar = np.sum(neg_log_sigmoid(fused[:n_tar]))
        loss_non = np.sum(neg_log_sigmoid(-fused[n_tar:]))

        # Total loss (normalized)
        return (loss_tar + loss_non) / (n_tar + n_non)

    # Initialize weights
    # Start with equal weights and logit(prior) offset
    from ..utils import logit
    w0 = np.ones(n_systems + 1) / n_systems
    w0[-1] = logit(prior)

    # Optimize using L-BFGS-B
    result = minimize(
        objective,
        w0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )

    if not result.success:
        warnings.warn(
            f"Linear fusion optimization did not converge: {result.message}"
        )

    weights = result.x

    # Create fusion function
    def fusion_func(scores: np.ndarray) -> np.ndarray:
        """Apply linear fusion to scores."""
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # Single system case, add dimension
            scores = scores.reshape(1, -1)
        # Compute: w^T @ scores + offset
        return weights[:-1] @ scores + weights[-1]

    return fusion_func, weights


def linear_fuse_scores(
    scores_list: List[Scores],
    key: Key,
    prior: float = 0.5,
    max_iter: int = 100
) -> Tuple[Scores, np.ndarray]:
    """
    Fuse multiple systems' scores using linear fusion.

    Trains fusion weights on the provided scores and applies them to
    produce fused scores. This is "cheating" fusion (train == test).

    Parameters
    ----------
    scores_list : list of Scores
        List of Scores objects, one per system to fuse
    key : Key
        Key object with target/non-target labels
    prior : float, optional
        Target prior for training (default: 0.5)
    max_iter : int, optional
        Maximum optimization iterations (default: 100)

    Returns
    -------
    fused_scores : Scores
        Fused scores object
    weights : ndarray
        Fusion weights of shape (n_systems + 1,)

    Examples
    --------
    >>> # Create scores for 2 systems
    >>> tar_mask = np.array([[True, False], [False, True]])
    >>> non_mask = np.array([[False, True], [True, False]])
    >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar_mask, non_mask)
    >>>
    >>> scores1 = Scores(['m1', 'm2'], ['t1', 't2'],
    ...                  np.array([[2.0, -1.0], [-0.5, 3.0]]))
    >>> scores2 = Scores(['m1', 'm2'], ['t1', 't2'],
    ...                  np.array([[1.5, -0.5], [0.0, 2.5]]))
    >>>
    >>> fused, weights = linear_fuse_scores([scores1, scores2], key)
    >>> print(f"Fusion weights: {weights}")

    Notes
    -----
    For proper evaluation, use linear_fuse_scores_dev_eval with separate
    development and evaluation sets.
    """
    if len(scores_list) == 0:
        raise ValueError("scores_list must not be empty")

    # Validate all scores
    for i, scores in enumerate(scores_list):
        scores.validate()

    key.validate()

    # Stack scores
    stacked = stack_scores(scores_list, key)

    # Get target and non-target scores
    tar_indices, non_indices = key.get_tar_non_arrays()

    tar_scores_stacked = stacked[:, tar_indices]
    non_scores_stacked = stacked[:, non_indices]

    # Train fusion
    fusion_func, weights = train_linear_fusion(
        tar_scores_stacked,
        non_scores_stacked,
        prior,
        max_iter
    )

    # Apply fusion to all scores
    fused_scores_array = fusion_func(stacked)

    # Create Scores object with fused scores
    # Reconstruct score matrix from flat array
    trial_mask = key.tar_mask | key.non_mask
    score_mat_fused = np.zeros(key.shape)
    score_mat_fused[trial_mask] = fused_scores_array

    fused_scores = Scores(
        key.model_names.copy(),
        key.test_names.copy(),
        score_mat_fused,
        trial_mask.copy()
    )

    return fused_scores, weights


def linear_fuse_scores_dev_eval(
    dev_scores_list: List[Scores],
    dev_key: Key,
    eval_scores_list: List[Scores],
    eval_key: Key,
    prior: float = 0.5,
    max_iter: int = 100
) -> Tuple[Scores, np.ndarray]:
    """
    Fuse scores using separate development and evaluation sets.

    Trains fusion weights on development data and applies them to
    evaluation data. This is the proper way to evaluate fusion performance.

    Parameters
    ----------
    dev_scores_list : list of Scores
        Development scores for training fusion
    dev_key : Key
        Development key with labels
    eval_scores_list : list of Scores
        Evaluation scores to fuse
    eval_key : Key
        Evaluation key
    prior : float, optional
        Target prior for training (default: 0.5)
    max_iter : int, optional
        Maximum optimization iterations (default: 100)

    Returns
    -------
    fused_eval_scores : Scores
        Fused evaluation scores
    weights : ndarray
        Fusion weights trained on development data

    Examples
    --------
    >>> # Development data
    >>> dev_tar_mask = np.array([[True, False]])
    >>> dev_non_mask = np.array([[False, True]])
    >>> dev_key = Key(['m1'], ['t1', 't2'], dev_tar_mask, dev_non_mask)
    >>> dev_scores1 = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
    >>> dev_scores2 = Scores(['m1'], ['t1', 't2'], np.array([[1.5, -0.5]]))
    >>>
    >>> # Evaluation data
    >>> eval_tar_mask = np.array([[True, False]])
    >>> eval_non_mask = np.array([[False, True]])
    >>> eval_key = Key(['m2'], ['t1', 't2'], eval_tar_mask, eval_non_mask)
    >>> eval_scores1 = Scores(['m2'], ['t1', 't2'], np.array([[1.8, -0.8]]))
    >>> eval_scores2 = Scores(['m2'], ['t1', 't2'], np.array([[1.2, -0.3]]))
    >>>
    >>> fused, weights = linear_fuse_scores_dev_eval(
    ...     [dev_scores1, dev_scores2], dev_key,
    ...     [eval_scores1, eval_scores2], eval_key
    ... )
    """
    if len(dev_scores_list) != len(eval_scores_list):
        raise ValueError(
            f"Number of systems must match: {len(dev_scores_list)} "
            f"dev vs {len(eval_scores_list)} eval"
        )

    if len(dev_scores_list) == 0:
        raise ValueError("scores_list must not be empty")

    # Validate
    dev_key.validate()
    eval_key.validate()

    # Stack development scores
    dev_stacked = stack_scores(dev_scores_list, dev_key)

    # Get development target and non-target scores
    tar_indices, non_indices = dev_key.get_tar_non_arrays()
    dev_tar_scores = dev_stacked[:, tar_indices]
    dev_non_scores = dev_stacked[:, non_indices]

    # Train fusion on development data
    fusion_func, weights = train_linear_fusion(
        dev_tar_scores,
        dev_non_scores,
        prior,
        max_iter
    )

    # Stack evaluation scores
    eval_stacked = stack_scores(eval_scores_list, eval_key)

    # Apply fusion to evaluation scores
    fused_eval_array = fusion_func(eval_stacked)

    # Create Scores object for fused evaluation scores
    trial_mask = eval_key.tar_mask | eval_key.non_mask
    score_mat_fused = np.zeros(eval_key.shape)
    score_mat_fused[trial_mask] = fused_eval_array

    fused_eval_scores = Scores(
        eval_key.model_names.copy(),
        eval_key.test_names.copy(),
        score_mat_fused,
        trial_mask.copy()
    )

    return fused_eval_scores, weights
