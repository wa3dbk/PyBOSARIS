"""
Evaluation metrics for PyBOSARIS.

This module provides performance evaluation metrics for binary classification
systems, particularly for speaker recognition and biometric systems.

Metrics:
    - EER (Equal Error Rate)
    - minDCF (minimum Detection Cost Function)
    - Cllr (log-likelihood ratio cost / calibration loss)
    - minCllr (minimum Cllr after optimal calibration)
    - ROCCH (ROC Convex Hull)
"""

import numpy as np
from typing import Tuple, Optional

from ..calibration.pav import pavx
from ..utils import sigmoid, neg_log_sigmoid, logit


def rocch(
    tar_scores: np.ndarray,
    non_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC Convex Hull (ROCCH).

    The ROCCH represents the best achievable performance for any monotonic
    transformation of the scores. It's computed using the Pool Adjacent
    Violators (PAV) algorithm.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)

    Returns
    -------
    pmiss : ndarray
        Miss probabilities at ROCCH vertices
    pfa : ndarray
        False alarm probabilities at ROCCH vertices

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> pmiss, pfa = rocch(tar, non)
    >>> # pmiss and pfa define the convex hull of the ROC curve

    Notes
    -----
    The ROCCH is always convex and monotonically decreasing.
    It represents the optimal achievable performance if scores can be
    transformed optimally (but monotonically).

    References
    ----------
    Based on the BOSARIS Toolkit ROCCH implementation.
    """
    tar_scores = np.asarray(tar_scores).ravel()
    non_scores = np.asarray(non_scores).ravel()

    n_tar = len(tar_scores)
    n_non = len(non_scores)
    n_total = n_tar + n_non

    # Combine all scores
    all_scores = np.concatenate([tar_scores, non_scores])

    # Ideal posterior: 1 for targets, 0 for non-targets
    ideal_posterior = np.concatenate([
        np.ones(n_tar),
        np.zeros(n_non)
    ])

    # Sort by score (stable sort to preserve order of ties)
    sorted_indices = np.argsort(all_scores, kind='stable')
    all_scores_sorted = all_scores[sorted_indices]
    ideal_posterior_sorted = ideal_posterior[sorted_indices]

    # Apply PAV to get optimal posterior probabilities
    optimal_posterior, width, height = pavx(ideal_posterior_sorted)

    # Compute miss and false alarm rates at each bin boundary
    n_bins = len(width)
    pmiss = np.zeros(n_bins + 1)
    pfa = np.zeros(n_bins + 1)

    # Threshold leftmost: accept everything, miss nothing
    left = 0  # Number of scores to left of threshold
    fa = n_non
    miss = 0

    for i in range(n_bins):
        pmiss[i] = miss / n_tar
        pfa[i] = fa / n_non

        left += width[i]
        miss = np.sum(ideal_posterior_sorted[:left])
        fa = n_total - left - np.sum(ideal_posterior_sorted[left:])

    # Final point: threshold rightmost
    pmiss[n_bins] = miss / n_tar
    pfa[n_bins] = fa / n_non

    return pmiss, pfa


def rocch2eer(pmiss: np.ndarray, pfa: np.ndarray) -> float:
    """
    Calculate Equal Error Rate (EER) from ROCCH coordinates.

    Parameters
    ----------
    pmiss : ndarray
        Miss probabilities at ROCCH vertices
    pfa : ndarray
        False alarm probabilities at ROCCH vertices

    Returns
    -------
    float
        Equal Error Rate (where Pmiss = Pfa)

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> pmiss, pfa = rocch(tar, non)
    >>> eer = rocch2eer(pmiss, pfa)

    Notes
    -----
    The EER is the point on the ROCCH where the miss rate equals
    the false alarm rate.
    """
    pmiss = np.asarray(pmiss)
    pfa = np.asarray(pfa)

    eer = 0.0

    # Iterate through each segment of the ROCCH
    for i in range(len(pfa) - 1):
        xx = pfa[i:i+2]
        yy = pmiss[i:i+2]

        # Check that segment is properly ordered
        # (pfa decreasing, pmiss increasing as we move along ROCCH)
        assert xx[1] <= xx[0] and yy[0] <= yy[1], \
            "ROCCH coordinates not properly ordered"

        # Create matrix of segment endpoints
        XY = np.column_stack([xx, yy])
        dd = np.array([1, -1]) @ XY

        if np.min(np.abs(dd)) == 0:
            # Segment already on the EER line
            eer_seg = 0
        else:
            # Find line coefficients such that seg^T [x; y] = 1
            # when (x, y) is on the line
            seg = np.linalg.solve(XY, np.ones(2))
            eer_seg = 1.0 / np.sum(seg)

        # EER is the highest candidate from all segments
        eer = max(eer, eer_seg)

    return eer


def compute_eer(tar_scores: np.ndarray, non_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) from scores.

    The EER is the operating point where the miss rate equals the false
    alarm rate. It's a commonly used metric for binary classification
    performance, particularly in biometric systems.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)

    Returns
    -------
    float
        Equal Error Rate (0 to 1, where lower is better)

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> eer = compute_eer(tar, non)
    >>> print(f"EER: {eer*100:.2f}%")

    Notes
    -----
    - Range: 0 <= EER <= 0.5 (50%)
    - EER = 0.5 means random performance
    - EER = 0 means perfect separation
    - The EER is computed using the ROCCH (convex hull)
    """
    pmiss, pfa = rocch(tar_scores, non_scores)
    return rocch2eer(pmiss, pfa)


def compute_min_dcf(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    prior: float = 0.5,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    normalize: bool = True
) -> Tuple[float, float, float]:
    """
    Compute minimum Detection Cost Function (minDCF).

    The minDCF is the minimum cost achievable by choosing an optimal
    decision threshold, given a prior and cost parameters.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)
    prior : float, optional
        Target prior probability (default: 0.5)
    c_miss : float, optional
        Cost of a miss (default: 1.0)
    c_fa : float, optional
        Cost of a false alarm (default: 1.0)
    normalize : bool, optional
        If True, normalize by min(P_tar * C_miss, (1-P_tar) * C_fa)
        (default: True)

    Returns
    -------
    min_dcf : float
        Minimum detection cost function
    pmiss_opt : float
        Miss probability at optimal threshold
    pfa_opt : float
        False alarm probability at optimal threshold

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> min_dcf, pmiss, pfa = compute_min_dcf(tar, non, prior=0.01)
    >>> print(f"minDCF: {min_dcf:.4f}")

    Notes
    -----
    The detection cost function is:
        DCF = P_tar * C_miss * P_miss(t) + (1 - P_tar) * C_fa * P_fa(t)

    where t is the decision threshold. The minDCF is the minimum over all t.

    If normalize=True, the result is divided by min(P_tar * C_miss, (1-P_tar) * C_fa),
    giving a normalized minDCF in the range [0, 1].
    """
    if not 0 < prior < 1:
        raise ValueError(f"prior must be in (0, 1), got {prior}")
    if c_miss <= 0:
        raise ValueError(f"c_miss must be positive, got {c_miss}")
    if c_fa <= 0:
        raise ValueError(f"c_fa must be positive, got {c_fa}")

    # Compute ROCCH
    pmiss, pfa = rocch(tar_scores, non_scores)

    # Compute cost at each ROCCH vertex
    p_tar = prior
    p_non = 1 - prior

    costs = p_tar * c_miss * pmiss + p_non * c_fa * pfa

    # Find minimum
    min_idx = np.argmin(costs)
    min_dcf = costs[min_idx]
    pmiss_opt = pmiss[min_idx]
    pfa_opt = pfa[min_idx]

    # Normalize if requested
    if normalize:
        norm_factor = min(p_tar * c_miss, p_non * c_fa)
        min_dcf = min_dcf / norm_factor

    return min_dcf, pmiss_opt, pfa_opt


def compute_cllr(tar_llrs: np.ndarray, non_llrs: np.ndarray) -> float:
    """
    Compute Cllr (log-likelihood ratio cost / calibration loss).

    Cllr measures both discrimination and calibration quality of LLR scores.
    It's the average negative log-likelihood (cross-entropy) using base-2
    logarithms, normalized to be 0 for perfect LLRs and 1 for LLR=0.

    Parameters
    ----------
    tar_llrs : ndarray
        Target trial log-likelihood-ratios (1D array)
    non_llrs : ndarray
        Non-target trial log-likelihood-ratios (1D array)

    Returns
    -------
    float
        Cllr value (0 to infinity, where lower is better)

    Examples
    --------
    >>> # Well-calibrated LLRs
    >>> tar_llrs = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non_llrs = np.array([-2.0, -1.0, -0.5, 0.0])
    >>> cllr = compute_cllr(tar_llrs, non_llrs)
    >>> print(f"Cllr: {cllr:.4f}")

    Notes
    -----
    Range: 0 <= Cllr <= infinity
    - Cllr = 0: Perfect calibration and discrimination
    - Cllr = 1: Performance of a system that outputs LLR=0 (no information)
    - Cllr > 1: Worse than providing no information (bad calibration)

    Cllr = (Cllr_tar + Cllr_non) / 2
    where:
        Cllr_tar = mean(-log2(sigmoid(tar_llrs)))
        Cllr_non = mean(-log2(sigmoid(-non_llrs)))

    References
    ----------
    Brummer & du Preez, "Application-Independent Evaluation of Speaker
    Detection", Computer Speech and Language, 2006.
    """
    tar_llrs = np.asarray(tar_llrs).ravel()
    non_llrs = np.asarray(non_llrs).ravel()

    if len(tar_llrs) == 0:
        raise ValueError("tar_llrs must not be empty")
    if len(non_llrs) == 0:
        raise ValueError("non_llrs must not be empty")

    # Cost for target trials: -log2(sigmoid(llr))
    # Use neg_log_sigmoid which is numerically stable
    c_tar = np.mean(neg_log_sigmoid(tar_llrs)) / np.log(2)

    # Cost for non-target trials: -log2(sigmoid(-llr))
    c_non = np.mean(neg_log_sigmoid(-non_llrs)) / np.log(2)

    # Total Cllr is average of both
    cllr = (c_tar + c_non) / 2

    return cllr


def compute_min_cllr(tar_scores: np.ndarray, non_scores: np.ndarray) -> float:
    """
    Compute minimum Cllr (minCllr).

    minCllr measures discrimination quality independent of calibration.
    It's the Cllr obtained after applying optimal PAV calibration to the scores.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)

    Returns
    -------
    float
        minCllr value (0 to 1, where lower is better)

    Examples
    --------
    >>> # Uncalibrated scores
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> min_cllr = compute_min_cllr(tar, non)
    >>> print(f"minCllr: {min_cllr:.4f}")

    Notes
    -----
    Range: 0 <= minCllr <= 1
    - minCllr = 0: Perfect discrimination
    - minCllr = 1: No discrimination (random performance)

    Unlike Cllr, minCllr cannot exceed 1 because it uses optimal calibration.

    The difference (Cllr - minCllr) measures calibration quality.

    References
    ----------
    Brummer & du Preez, "Application-Independent Evaluation of Speaker
    Detection", Computer Speech and Language, 2006.
    """
    tar_scores = np.asarray(tar_scores).ravel()
    non_scores = np.asarray(non_scores).ravel()

    if len(tar_scores) == 0:
        raise ValueError("tar_scores must not be empty")
    if len(non_scores) == 0:
        raise ValueError("non_scores must not be empty")

    n_tar = len(tar_scores)
    n_non = len(non_scores)
    n_total = n_tar + n_non

    # Combine all scores
    all_scores = np.concatenate([tar_scores, non_scores])

    # Ideal posterior: 1 for targets, 0 for non-targets
    ideal_posterior = np.concatenate([
        np.ones(n_tar),
        np.zeros(n_non)
    ])

    # Sort by score
    sorted_indices = np.argsort(all_scores, kind='stable')
    all_scores_sorted = all_scores[sorted_indices]
    ideal_posterior_sorted = ideal_posterior[sorted_indices]

    # Apply PAV to get optimal posterior probabilities
    optimal_posterior, _, _ = pavx(ideal_posterior_sorted)

    # Compute data prior
    data_prior = n_tar / n_total

    # Convert optimal posteriors to LLRs
    # LLR = log(P(target|score) / P(nontarget|score)) - log(P(target) / P(nontarget))
    # This removes the effect of the data prior
    with np.errstate(divide='ignore', invalid='ignore'):
        optimal_llrs = logit(optimal_posterior) - logit(data_prior)

    # Clip infinities
    optimal_llrs = np.clip(optimal_llrs, -1e6, 1e6)

    # Split back into target and non-target LLRs
    # Need to undo the sorting
    inverse_indices = np.argsort(sorted_indices)
    optimal_llrs_original_order = optimal_llrs[inverse_indices]

    tar_llrs = optimal_llrs_original_order[:n_tar]
    non_llrs = optimal_llrs_original_order[n_tar:]

    # Compute Cllr on optimally calibrated scores
    return compute_cllr(tar_llrs, non_llrs)


def compute_act_dcf(
    tar_llrs: np.ndarray,
    non_llrs: np.ndarray,
    prior: float = 0.5,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    normalize: bool = True
) -> Tuple[float, float, float]:
    """
    Compute actual Detection Cost Function (actDCF) at Bayes threshold.

    The actDCF is the empirical cost when using log-likelihood-ratio scores
    with the Bayes decision threshold. Unlike minDCF which optimizes over
    all thresholds, actDCF uses the theoretically optimal threshold.

    Parameters
    ----------
    tar_llrs : ndarray
        Target trial log-likelihood-ratios (1D array)
    non_llrs : ndarray
        Non-target trial log-likelihood-ratios (1D array)
    prior : float, optional
        Target prior probability (default: 0.5)
    c_miss : float, optional
        Cost of a miss (default: 1.0)
    c_fa : float, optional
        Cost of a false alarm (default: 1.0)
    normalize : bool, optional
        If True, normalize by min(P_tar * C_miss, (1-P_tar) * C_fa)
        (default: True)

    Returns
    -------
    act_dcf : float
        Actual detection cost function at Bayes threshold
    pmiss : float
        Empirical miss rate at Bayes threshold
    pfa : float
        Empirical false alarm rate at Bayes threshold

    Examples
    --------
    >>> # Well-calibrated LLRs
    >>> tar_llrs = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non_llrs = np.array([-2.0, -1.0, -0.5, 0.0])
    >>> act_dcf, pmiss, pfa = compute_act_dcf(tar_llrs, non_llrs, prior=0.01)
    >>> print(f"actDCF: {act_dcf:.4f}")

    Notes
    -----
    The Bayes decision threshold for LLRs is:
        threshold = -log(P_tar * C_miss / ((1-P_tar) * C_fa))
                  = -logit(effective_prior)

    The decision rule is: accept if LLR >= threshold, reject otherwise.

    The actDCF measures calibration quality: if scores are well-calibrated
    LLRs, actDCF should be close to minDCF. The ratio (actDCF / minDCF)
    or the difference (actDCF - minDCF) quantifies calibration loss.
    """
    tar_llrs = np.asarray(tar_llrs).ravel()
    non_llrs = np.asarray(non_llrs).ravel()

    if len(tar_llrs) == 0:
        raise ValueError("tar_llrs must not be empty")
    if len(non_llrs) == 0:
        raise ValueError("non_llrs must not be empty")

    if not 0 < prior < 1:
        raise ValueError(f"prior must be in (0, 1), got {prior}")
    if c_miss <= 0:
        raise ValueError(f"c_miss must be positive, got {c_miss}")
    if c_fa <= 0:
        raise ValueError(f"c_fa must be positive, got {c_fa}")

    # Compute effective prior
    from ..utils import effective_prior as compute_eff_prior, logit
    eff_prior = compute_eff_prior(prior, c_miss, c_fa)

    # Bayes threshold = -logit(effective_prior)
    threshold = -logit(eff_prior)

    # Count misses and false alarms at this threshold
    # Decision rule: accept if LLR >= threshold
    n_tar = len(tar_llrs)
    n_non = len(non_llrs)

    # Targets with LLR < threshold are misses
    n_miss = np.sum(tar_llrs < threshold)
    pmiss = n_miss / n_tar

    # Non-targets with LLR >= threshold are false alarms
    n_fa = np.sum(non_llrs >= threshold)
    pfa = n_fa / n_non

    # Compute cost
    p_tar = prior
    p_non = 1 - prior
    act_dcf = p_tar * c_miss * pmiss + p_non * c_fa * pfa

    # Normalize if requested
    if normalize:
        norm_factor = min(p_tar * c_miss, p_non * c_fa)
        act_dcf = act_dcf / norm_factor

    return act_dcf, pmiss, pfa


def compute_all_metrics(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    prior: float = 0.5,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    normalize_dcf: bool = True,
    is_llr: bool = False
) -> dict:
    """
    Compute all common evaluation metrics.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores (1D array)
    non_scores : ndarray
        Non-target trial scores (1D array)
    prior : float, optional
        Target prior for DCF computation (default: 0.5)
    c_miss : float, optional
        Cost of miss for DCF (default: 1.0)
    c_fa : float, optional
        Cost of false alarm for DCF (default: 1.0)
    normalize_dcf : bool, optional
        Whether to normalize DCF (default: True)
    is_llr : bool, optional
        If True, scores are already log-likelihood-ratios (default: False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'eer': Equal Error Rate
        - 'min_dcf': Minimum Detection Cost Function
        - 'pmiss_opt': Miss rate at optimal threshold for DCF
        - 'pfa_opt': False alarm rate at optimal threshold for DCF
        - 'cllr': Calibration loss (only if is_llr=True)
        - 'min_cllr': Minimum calibration loss

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 3.5, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0, 1.0])
    >>> metrics = compute_all_metrics(tar, non, prior=0.01)
    >>> print(f"EER: {metrics['eer']*100:.2f}%")
    >>> print(f"minDCF: {metrics['min_dcf']:.4f}")
    >>> print(f"minCllr: {metrics['min_cllr']:.4f}")
    """
    results = {}

    # EER
    results['eer'] = compute_eer(tar_scores, non_scores)

    # minDCF
    min_dcf, pmiss, pfa = compute_min_dcf(
        tar_scores, non_scores, prior, c_miss, c_fa, normalize_dcf
    )
    results['min_dcf'] = min_dcf
    results['pmiss_opt'] = pmiss
    results['pfa_opt'] = pfa

    # Cllr and actDCF (only if scores are LLRs)
    if is_llr:
        results['cllr'] = compute_cllr(tar_scores, non_scores)
        act_dcf, pmiss_act, pfa_act = compute_act_dcf(
            tar_scores, non_scores, prior, c_miss, c_fa, normalize_dcf
        )
        results['act_dcf'] = act_dcf
        results['pmiss_act'] = pmiss_act
        results['pfa_act'] = pfa_act

    # minCllr
    results['min_cllr'] = compute_min_cllr(tar_scores, non_scores)

    return results
