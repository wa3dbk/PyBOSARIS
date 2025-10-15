"""
Mathematical utility functions for PyBOSARIS.

This module provides basic mathematical transformations commonly used
in the BOSARIS toolkit, particularly for working with probabilities and
log-likelihood-ratios.
"""

import numpy as np
from typing import Union

# Type alias for array-like inputs
ArrayLike = Union[float, np.ndarray]


def logit(p: ArrayLike) -> ArrayLike:
    """
    Compute log-odds (logit) transformation.

    Maps probabilities from [0, 1] to log-odds in (-∞, ∞):
        logit(p) = log(p / (1 - p))

    Parameters
    ----------
    p : float or array_like
        Probability value(s) in [0, 1]

    Returns
    -------
    float or ndarray
        Log-odds value(s)

    Examples
    --------
    >>> logit(0.5)
    0.0
    >>> logit(0.9)
    2.197...
    >>> logit(np.array([0.1, 0.5, 0.9]))
    array([-2.197...,  0.   ,  2.197...])

    Notes
    -----
    - logit(0) = -∞
    - logit(1) = ∞
    - logit(0.5) = 0
    """
    p = np.asarray(p)
    with np.errstate(divide='ignore'):
        return np.log(p / (1 - p))


def sigmoid(x: ArrayLike) -> ArrayLike:
    """
    Compute logistic sigmoid (inverse logit) transformation.

    Maps log-odds from (-∞, ∞) to probabilities in [0, 1]:
        sigmoid(x) = 1 / (1 + exp(-x))

    Parameters
    ----------
    x : float or array_like
        Log-odds value(s)

    Returns
    -------
    float or ndarray
        Probability value(s) in [0, 1]

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(2.197)
    0.9...
    >>> sigmoid(np.array([-2.197, 0.0, 2.197]))
    array([0.1..., 0.5, 0.9...])

    Notes
    -----
    - sigmoid(-∞) = 0
    - sigmoid(∞) = 1
    - sigmoid(0) = 0.5
    - sigmoid is the inverse of logit: sigmoid(logit(p)) = p

    This implementation is numerically stable for large |x|.
    """
    x = np.asarray(x)
    # Numerically stable implementation
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def probit(p: ArrayLike) -> ArrayLike:
    """
    Compute probit transformation (inverse CDF of standard normal).

    Maps probabilities from [0, 1] to (-∞, ∞).
    Used for DET curve axis scaling.

    Parameters
    ----------
    p : float or array_like
        Probability value(s) in [0, 1]

    Returns
    -------
    float or ndarray
        Probit-transformed value(s)

    Examples
    --------
    >>> probit(0.5)
    0.0
    >>> probit(0.841...)
    1.0
    >>> probit(np.array([0.159, 0.5, 0.841]))
    array([-1.,  0.,  1.])

    Notes
    -----
    - probit(p) = √2 * erf⁻¹(2p - 1)
    - probit(0.5) = 0
    - Used for axis scaling in DET plots
    """
    from scipy.special import ndtri
    return ndtri(p)


def inv_probit(x: ArrayLike) -> ArrayLike:
    """
    Compute inverse probit transformation (CDF of standard normal).

    Maps (-∞, ∞) to probabilities in [0, 1].

    Parameters
    ----------
    x : float or array_like
        Probit value(s)

    Returns
    -------
    float or ndarray
        Probability value(s) in [0, 1]

    Examples
    --------
    >>> inv_probit(0.0)
    0.5
    >>> inv_probit(1.0)
    0.841...
    >>> inv_probit(np.array([-1.0, 0.0, 1.0]))
    array([0.159, 0.5, 0.841])

    Notes
    -----
    - inv_probit is the inverse of probit: inv_probit(probit(p)) = p
    - Also known as the normal CDF or Φ(x)
    """
    from scipy.special import ndtr
    return ndtr(x)


def effective_prior(
    target_prior: float,
    cost_miss: float = 1.0,
    cost_fa: float = 1.0
) -> float:
    """
    Calculate effective prior from target prior and costs.

    The effective prior combines the target prior and cost ratio into a single
    parameter that determines the Bayes decision threshold:

        π̃ = (π * C_miss) / (π * C_miss + (1 - π) * C_fa)

    Parameters
    ----------
    target_prior : float
        Prior probability of target hypothesis, π ∈ [0, 1]
    cost_miss : float, optional
        Cost of a miss (default: 1.0)
    cost_fa : float, optional
        Cost of a false alarm (default: 1.0)

    Returns
    -------
    float
        Effective prior π̃ ∈ [0, 1]

    Examples
    --------
    >>> effective_prior(0.01, cost_miss=10, cost_fa=1)
    0.0909...
    >>> effective_prior(0.5, cost_miss=1, cost_fa=1)
    0.5

    Notes
    -----
    The Bayes decision threshold can be computed from the effective prior:
        η = -logit(π̃)

    This simplifies evaluation across different cost/prior combinations.

    References
    ----------
    See BOSARIS Toolkit User Guide, Section 2.4.2
    """
    if not 0 <= target_prior <= 1:
        raise ValueError(f"target_prior must be in [0, 1], got {target_prior}")
    if cost_miss <= 0:
        raise ValueError(f"cost_miss must be positive, got {cost_miss}")
    if cost_fa <= 0:
        raise ValueError(f"cost_fa must be positive, got {cost_fa}")

    numerator = target_prior * cost_miss
    denominator = target_prior * cost_miss + (1 - target_prior) * cost_fa

    return numerator / denominator


def neg_log_sigmoid(x: ArrayLike) -> ArrayLike:
    """
    Compute -log(sigmoid(x)) in a numerically stable way.

    This is equivalent to log(1 + exp(-x)) but more stable.
    Used in logistic regression objective functions.

    Parameters
    ----------
    x : float or array_like
        Input value(s)

    Returns
    -------
    float or ndarray
        -log(sigmoid(x))

    Examples
    --------
    >>> neg_log_sigmoid(0.0)
    0.693...
    >>> neg_log_sigmoid(10.0)
    4.5...e-05

    Notes
    -----
    Numerically equivalent to but more stable than:
        -np.log(sigmoid(x))

    For x >= 0: log(1 + exp(-x))
    For x < 0: -x + log(1 + exp(x))  (to avoid exp(-x) which would be large)
    """
    x = np.asarray(x)
    return np.where(
        x >= 0,
        np.log1p(np.exp(-x)),        # For x >= 0
        -x + np.log1p(np.exp(x))     # For x < 0: mathematically equivalent
    )


def safe_log(x: ArrayLike, eps: float = 1e-10) -> ArrayLike:
    """
    Compute log with clipping to avoid log(0).

    Parameters
    ----------
    x : float or array_like
        Input value(s)
    eps : float, optional
        Minimum value before taking log (default: 1e-10)

    Returns
    -------
    float or ndarray
        log(max(x, eps))

    Examples
    --------
    >>> safe_log(1.0)
    0.0
    >>> safe_log(0.0)
    -23.025...
    >>> safe_log(np.array([0, 1, 10]))
    array([-23.025...,   0.   ,   2.302...])
    """
    x = np.asarray(x)
    return np.log(np.maximum(x, eps))
