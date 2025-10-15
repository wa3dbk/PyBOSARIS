"""
DET (Detection Error Tradeoff) curve plotting for PyBOSARIS.

DET curves visualize the tradeoff between miss rate and false alarm rate
in binary classification systems.

Note: Requires matplotlib package. Install with: pip install matplotlib
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from scipy.stats import norm

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    matplotlib = None


def _check_matplotlib():
    """Check if matplotlib is available and raise helpful error if not."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def _probit(p: np.ndarray) -> np.ndarray:
    """
    Compute probit (inverse normal CDF) for DET scale.

    Parameters
    ----------
    p : ndarray
        Probabilities (must be in (0, 1))

    Returns
    -------
    ndarray
        Probit values
    """
    # Clip to avoid infinities
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return norm.ppf(p)


def _format_det_axis(ax, axis: str = 'both'):
    """
    Format axis ticks for DET plot (probit scale with percentage labels).

    Parameters
    ----------
    ax : matplotlib axis
        Axis to format
    axis : str
        Which axis to format ('x', 'y', or 'both')
    """
    # Standard tick positions for DET plots
    ticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40])
    tick_positions = _probit(ticks / 100)
    tick_labels = [f'{t:.2g}' if t < 1 else f'{int(t)}' for t in ticks]

    if axis in ('x', 'both'):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('False Alarm Rate (%)')
        ax.grid(True, which='both', axis='x', alpha=0.3)

    if axis in ('y', 'both'):
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel('Miss Rate (%)')
        ax.grid(True, which='both', axis='y', alpha=0.3)


def plot_det_curve(
    tar_scores: np.ndarray,
    non_scores: np.ndarray,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: str = '-',
    linewidth: float = 2.0,
    ax: Optional['matplotlib.axes.Axes'] = None,
    plot_rocch: bool = False
) -> 'matplotlib.axes.Axes':
    """
    Plot DET (Detection Error Tradeoff) curve.

    Parameters
    ----------
    tar_scores : ndarray
        Target trial scores
    non_scores : ndarray
        Non-target trial scores
    label : str, optional
        Label for the curve (for legend)
    color : str, optional
        Line color
    linestyle : str, optional
        Line style (default: '-')
    linewidth : float, optional
        Line width (default: 2.0)
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure
    plot_rocch : bool, optional
        If True, plot ROC Convex Hull instead of raw DET (default: False)

    Returns
    -------
    ax : matplotlib axis
        Axis with DET curve

    Examples
    --------
    >>> tar = np.array([2.0, 3.0, 4.0])
    >>> non = np.array([-2.0, -1.0, 0.0])
    >>> ax = plot_det_curve(tar, non, label='System 1')
    >>> plt.show()

    Notes
    -----
    - DET curves use probit scale on both axes
    - Lower curves indicate better performance
    - The diagonal line represents random performance
    """
    _check_matplotlib()

    from ..evaluation import rocch

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Compute error rates
    if plot_rocch:
        # Use ROCCH for optimal performance curve
        pmiss, pfa = rocch(tar_scores, non_scores)
    else:
        # Compute empirical DET curve
        # Combine and sort scores
        all_scores = np.concatenate([tar_scores, non_scores])
        is_target = np.concatenate([np.ones(len(tar_scores)), np.zeros(len(non_scores))])

        # Sort by score (descending)
        sorted_idx = np.argsort(-all_scores)
        sorted_scores = all_scores[sorted_idx]
        sorted_labels = is_target[sorted_idx]

        # Compute cumulative error rates at each threshold
        n_tar = len(tar_scores)
        n_non = len(non_scores)

        # Start with all rejected (pmiss=1, pfa=0)
        pmiss_list = [1.0]
        pfa_list = [0.0]

        cumsum_tar = 0
        cumsum_non = 0

        for i in range(len(sorted_scores)):
            if sorted_labels[i] == 1:
                cumsum_tar += 1
            else:
                cumsum_non += 1

            # Miss rate: fraction of targets not yet accepted
            pmiss = 1.0 - (cumsum_tar / n_tar)
            # False alarm rate: fraction of non-targets accepted
            pfa = cumsum_non / n_non

            pmiss_list.append(pmiss)
            pfa_list.append(pfa)

        # End with all accepted (pmiss=0, pfa=1)
        pmiss_list.append(0.0)
        pfa_list.append(1.0)

        pmiss = np.array(pmiss_list)
        pfa = np.array(pfa_list)

    # Convert to probit scale
    pmiss_probit = _probit(pmiss)
    pfa_probit = _probit(pfa)

    # Plot curve
    ax.plot(pfa_probit, pmiss_probit,
            label=label, color=color, linestyle=linestyle, linewidth=linewidth)

    # Format axes
    _format_det_axis(ax, axis='both')

    # Add diagonal line for reference (random performance)
    ax_limits = ax.axis()
    diag_min = max(ax_limits[0], ax_limits[2])
    diag_max = min(ax_limits[1], ax_limits[3])
    ax.plot([diag_min, diag_max], [diag_min, diag_max],
            'k--', alpha=0.3, linewidth=1, label='Random' if label else None)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add legend if labels exist
    if label:
        ax.legend(loc='best')

    ax.set_title('DET Curve')

    return ax


def plot_det_curves(
    scores_list: List[Tuple[np.ndarray, np.ndarray]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = 'DET Curves',
    figsize: Tuple[float, float] = (10, 10),
    plot_rocch: bool = False
) -> 'matplotlib.axes.Axes':
    """
    Plot multiple DET curves on same axes.

    Parameters
    ----------
    scores_list : list of tuple
        List of (tar_scores, non_scores) tuples
    labels : list of str, optional
        Labels for each curve
    colors : list of str, optional
        Colors for each curve
    title : str, optional
        Plot title (default: 'DET Curves')
    figsize : tuple, optional
        Figure size (default: (10, 10))
    plot_rocch : bool, optional
        If True, plot ROC Convex Hull curves (default: False)

    Returns
    -------
    ax : matplotlib axis
        Axis with all DET curves

    Examples
    --------
    >>> tar1 = np.array([2.0, 3.0, 4.0])
    >>> non1 = np.array([-2.0, -1.0, 0.0])
    >>> tar2 = np.array([1.5, 2.5, 3.5])
    >>> non2 = np.array([-1.5, -0.5, 0.5])
    >>> ax = plot_det_curves(
    ...     [(tar1, non1), (tar2, non2)],
    ...     labels=['System 1', 'System 2']
    ... )
    >>> plt.show()
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        labels = [None] * len(scores_list)
    if colors is None:
        colors = [None] * len(scores_list)

    for (tar_scores, non_scores), label, color in zip(scores_list, labels, colors):
        plot_det_curve(
            tar_scores, non_scores,
            label=label, color=color, ax=ax, plot_rocch=plot_rocch
        )

    ax.set_title(title)

    return ax


def plot_det_from_scores(
    scores,
    key,
    label: Optional[str] = None,
    color: Optional[str] = None,
    ax: Optional['matplotlib.axes.Axes'] = None,
    plot_rocch: bool = False
) -> 'matplotlib.axes.Axes':
    """
    Plot DET curve from Scores and Key objects.

    Parameters
    ----------
    scores : Scores
        Scores object
    key : Key
        Key object with target/non-target labels
    label : str, optional
        Label for the curve
    color : str, optional
        Line color
    ax : matplotlib axis, optional
        Axis to plot on
    plot_rocch : bool, optional
        If True, plot ROC Convex Hull (default: False)

    Returns
    -------
    ax : matplotlib axis
        Axis with DET curve

    Examples
    --------
    >>> from pybosaris.core import Scores, Key
    >>> tar_mask = np.array([[True, False]])
    >>> non_mask = np.array([[False, True]])
    >>> key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)
    >>> scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
    >>> ax = plot_det_from_scores(scores, key, label='My System')
    >>> plt.show()
    """
    _check_matplotlib()

    from ..core import Scores, Key

    if not isinstance(scores, Scores):
        raise TypeError("scores must be a Scores object")
    if not isinstance(key, Key):
        raise TypeError("key must be a Key object")

    # Extract target and non-target scores
    tar_scores, non_scores = scores.get_tar_non(key)

    return plot_det_curve(
        tar_scores, non_scores,
        label=label, color=color, ax=ax, plot_rocch=plot_rocch
    )


def save_det_plot(
    filename: str,
    dpi: int = 300,
    bbox_inches: str = 'tight'
):
    """
    Save current DET plot to file.

    Parameters
    ----------
    filename : str
        Output filename (extension determines format: .png, .pdf, .svg, etc.)
    dpi : int, optional
        Resolution for raster formats (default: 300)
    bbox_inches : str, optional
        Bounding box setting (default: 'tight')

    Examples
    --------
    >>> plot_det_curve(tar, non, label='System')
    >>> save_det_plot('det_curve.png')
    """
    _check_matplotlib()

    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
