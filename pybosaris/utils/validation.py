"""
Validation functions for PyBOSARIS.

This module provides functions for validating input data structures and parameters.
"""

import numpy as np
from typing import Optional, Tuple, List


def validate_scores(
    scores: np.ndarray,
    allow_nan: bool = True,
    allow_inf: bool = False
) -> None:
    """
    Validate score array.

    Parameters
    ----------
    scores : ndarray
        Array of scores to validate
    allow_nan : bool, optional
        Whether to allow NaN values (default: True, for missing scores)
    allow_inf : bool, optional
        Whether to allow infinite values (default: False)

    Raises
    ------
    TypeError
        If scores is not a numpy array
    ValueError
        If scores contain invalid values

    Examples
    --------
    >>> scores = np.array([1.0, 2.0, 3.0])
    >>> validate_scores(scores)  # No error

    >>> scores_with_nan = np.array([1.0, np.nan, 3.0])
    >>> validate_scores(scores_with_nan, allow_nan=True)  # No error

    >>> scores_with_inf = np.array([1.0, np.inf, 3.0])
    >>> validate_scores(scores_with_inf)  # Raises ValueError
    """
    if not isinstance(scores, np.ndarray):
        raise TypeError(f"Scores must be numpy array, got {type(scores)}")

    if not allow_nan and np.any(np.isnan(scores)):
        raise ValueError("Scores contain NaN values")

    if not allow_inf and np.any(np.isinf(scores)):
        raise ValueError("Scores contain infinite values")


def validate_labels(
    labels: np.ndarray,
    allow_values: Optional[List] = None
) -> None:
    """
    Validate label array.

    Parameters
    ----------
    labels : ndarray
        Array of labels to validate
    allow_values : list, optional
        List of allowed label values (default: None, allows any)

    Raises
    ------
    TypeError
        If labels is not a numpy array
    ValueError
        If labels contain invalid values

    Examples
    --------
    >>> labels = np.array([0, 1, 1, 0])
    >>> validate_labels(labels, allow_values=[0, 1])  # No error

    >>> labels = np.array([0, 1, 2])
    >>> validate_labels(labels, allow_values=[0, 1])  # Raises ValueError
    """
    if not isinstance(labels, np.ndarray):
        raise TypeError(f"Labels must be numpy array, got {type(labels)}")

    if allow_values is not None:
        unique_values = np.unique(labels[~np.isnan(labels)])
        invalid = set(unique_values) - set(allow_values)
        if invalid:
            raise ValueError(f"Labels contain invalid values: {invalid}")


def validate_trial_mask(
    mask: np.ndarray,
    expected_shape: Optional[Tuple[int, int]] = None
) -> None:
    """
    Validate trial mask (boolean array indicating valid trials).

    Parameters
    ----------
    mask : ndarray
        Boolean mask array
    expected_shape : tuple, optional
        Expected shape (n_models, n_test_segments)

    Raises
    ------
    TypeError
        If mask is not a numpy array or not boolean
    ValueError
        If mask has incorrect shape

    Examples
    --------
    >>> mask = np.array([[True, False], [False, True]])
    >>> validate_trial_mask(mask, expected_shape=(2, 2))  # No error

    >>> mask = np.array([[1, 0], [0, 1]])
    >>> validate_trial_mask(mask)  # Raises TypeError (not boolean)
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Mask must be numpy array, got {type(mask)}")

    if mask.dtype != bool:
        raise TypeError(f"Mask must be boolean array, got dtype {mask.dtype}")

    if expected_shape is not None:
        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask has shape {mask.shape}, expected {expected_shape}"
            )


def validate_key_consistency(
    tar_mask: np.ndarray,
    non_mask: np.ndarray
) -> None:
    """
    Validate that target and non-target masks don't overlap.

    Parameters
    ----------
    tar_mask : ndarray
        Boolean mask for target trials
    non_mask : ndarray
        Boolean mask for non-target trials

    Raises
    ------
    ValueError
        If masks overlap (same trial marked as both target and non-target)

    Examples
    --------
    >>> tar = np.array([[True, False], [False, False]])
    >>> non = np.array([[False, True], [True, False]])
    >>> validate_key_consistency(tar, non)  # No error

    >>> tar = np.array([[True, False], [False, False]])
    >>> non = np.array([[True, True], [False, False]])
    >>> validate_key_consistency(tar, non)  # Raises ValueError
    """
    if tar_mask.shape != non_mask.shape:
        raise ValueError(
            f"Target and non-target masks have different shapes: "
            f"{tar_mask.shape} vs {non_mask.shape}"
        )

    overlap = np.logical_and(tar_mask, non_mask)
    if np.any(overlap):
        n_overlap = np.sum(overlap)
        raise ValueError(
            f"Target and non-target masks overlap at {n_overlap} positions"
        )


def validate_probability(
    p: float,
    param_name: str = "probability"
) -> None:
    """
    Validate that a value is a valid probability in [0, 1].

    Parameters
    ----------
    p : float
        Probability value to validate
    param_name : str, optional
        Name of parameter for error messages

    Raises
    ------
    TypeError
        If p is not a number
    ValueError
        If p is not in [0, 1]

    Examples
    --------
    >>> validate_probability(0.5, "prior")  # No error
    >>> validate_probability(1.5, "prior")  # Raises ValueError
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        raise TypeError(f"{param_name} must be a number, got {type(p)}")

    if not 0 <= p <= 1:
        raise ValueError(f"{param_name} must be in [0, 1], got {p}")


def validate_positive(
    value: float,
    param_name: str = "value"
) -> None:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float
        Value to validate
    param_name : str, optional
        Name of parameter for error messages

    Raises
    ------
    TypeError
        If value is not a number
    ValueError
        If value is not positive

    Examples
    --------
    >>> validate_positive(1.0, "cost")  # No error
    >>> validate_positive(-1.0, "cost")  # Raises ValueError
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{param_name} must be a number, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")


def validate_model_test_names(
    model_names: List[str],
    test_names: List[str]
) -> None:
    """
    Validate model and test segment name lists.

    Parameters
    ----------
    model_names : list of str
        Model name list
    test_names : list of str
        Test segment name list

    Raises
    ------
    TypeError
        If names are not lists of strings
    ValueError
        If there are duplicate names

    Examples
    --------
    >>> validate_model_test_names(['m1', 'm2'], ['t1', 't2'])  # No error
    >>> validate_model_test_names(['m1', 'm1'], ['t1', 't2'])  # Raises ValueError
    """
    if not isinstance(model_names, list):
        raise TypeError(f"model_names must be list, got {type(model_names)}")

    if not isinstance(test_names, list):
        raise TypeError(f"test_names must be list, got {type(test_names)}")

    if not all(isinstance(name, str) for name in model_names):
        raise TypeError("All model names must be strings")

    if not all(isinstance(name, str) for name in test_names):
        raise TypeError("All test names must be strings")

    # Check for duplicates
    if len(model_names) != len(set(model_names)):
        dups = [name for name in set(model_names) if model_names.count(name) > 1]
        raise ValueError(f"Duplicate model names: {dups}")

    if len(test_names) != len(set(test_names)):
        dups = [name for name in set(test_names) if test_names.count(name) > 1]
        raise ValueError(f"Duplicate test names: {dups}")


def check_sufficient_trials(
    n_target: int,
    n_nontarget: int,
    min_trials: int = 30,
    warn: bool = True
) -> Tuple[bool, bool]:
    """
    Check if there are sufficient trials for reliable evaluation.

    Based on Doddington's Rule of 30: need at least 30 errors of each type
    for reliable error rate estimation.

    Parameters
    ----------
    n_target : int
        Number of target trials
    n_nontarget : int
        Number of non-target trials
    min_trials : int, optional
        Minimum number of trials recommended (default: 30)
    warn : bool, optional
        Whether to print warnings (default: True)

    Returns
    -------
    sufficient_targets : bool
        Whether there are enough target trials
    sufficient_nontargets : bool
        Whether there are enough non-target trials

    Examples
    --------
    >>> check_sufficient_trials(100, 1000, min_trials=30)
    (True, True)

    >>> check_sufficient_trials(10, 1000, min_trials=30)
    (False, True)

    Notes
    -----
    Doddington's Rule of 30 suggests you need at least 30 errors to get
    a probably approximately correct error-rate estimate.

    References
    ----------
    Doddington, G. R. (1998). "Speaker recognition evaluation methodology."
    """
    sufficient_targets = n_target >= min_trials
    sufficient_nontargets = n_nontarget >= min_trials

    if warn:
        if not sufficient_targets:
            print(
                f"Warning: Only {n_target} target trials "
                f"(recommended: >= {min_trials})"
            )
        if not sufficient_nontargets:
            print(
                f"Warning: Only {n_nontarget} non-target trials "
                f"(recommended: >= {min_trials})"
            )

    return sufficient_targets, sufficient_nontargets
