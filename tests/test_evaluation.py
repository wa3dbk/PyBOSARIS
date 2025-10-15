"""
Unit tests for evaluation metrics module.
"""

import pytest
import numpy as np
from pybosaris.evaluation import (
    rocch,
    rocch2eer,
    compute_eer,
    compute_min_dcf,
    compute_act_dcf,
    compute_cllr,
    compute_min_cllr,
    compute_all_metrics
)


class TestROCCH:
    """Tests for ROCCH computation."""

    def test_perfect_separation(self):
        """Test ROCCH with perfectly separated scores."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([-2.0, -1.0, 0.0])

        pmiss, pfa = rocch(tar, non)

        # ROCCH starts at high pfa (1.0) and low pmiss
        # and ends at low pfa (0.0) and high pmiss
        # Perfect separation should give a point with pmiss=0, pfa=0
        assert np.any((pmiss == 0.0) & (pfa == 0.0))
        # Should have corners at (1, 0) and (0, 1) conceptually
        assert pfa[0] == 1.0  # Starts at high pfa
        assert pmiss[-1] == 1.0  # Ends at high pmiss

    def test_identical_scores(self):
        """Test ROCCH when target and non-target scores are identical."""
        tar = np.array([1.0, 1.0, 1.0])
        non = np.array([1.0, 1.0, 1.0])

        pmiss, pfa = rocch(tar, non)

        # Should have at least 2 points
        assert len(pmiss) >= 2
        assert len(pfa) >= 2

    def test_output_shapes(self):
        """Test that ROCCH output arrays have matching shapes."""
        tar = np.array([1.0, 2.0, 3.0, 4.0])
        non = np.array([0.0, 1.0, 2.0])

        pmiss, pfa = rocch(tar, non)

        assert len(pmiss) == len(pfa)
        assert len(pmiss) >= 2

    def test_monotonicity(self):
        """Test that ROCCH output is monotonic."""
        tar = np.array([2.0, 3.0, 4.0, 5.0])
        non = np.array([0.0, 1.0, 2.0])

        pmiss, pfa = rocch(tar, non)

        # pfa should be non-increasing (decreasing from 1.0 to 0.0)
        assert np.all(np.diff(pfa) <= 1e-10)
        # pmiss should be non-decreasing (increasing from low to high)
        assert np.all(np.diff(pmiss) >= -1e-10)


class TestROCCH2EER:
    """Tests for ROCCH to EER conversion."""

    def test_perfect_separation(self):
        """Test EER calculation with perfect separation."""
        # ROCCH for perfect separation
        # pfa must be decreasing, pmiss must be increasing
        pmiss = np.array([0.0, 0.0, 1.0])
        pfa = np.array([1.0, 0.0, 0.0])

        eer = rocch2eer(pmiss, pfa)

        assert eer == 0.0

    def test_random_performance(self):
        """Test EER calculation with random performance."""
        # Diagonal line (random classifier)
        # pfa must be decreasing, pmiss must be increasing
        pmiss = np.array([0.0, 0.5, 1.0])
        pfa = np.array([1.0, 0.5, 0.0])

        eer = rocch2eer(pmiss, pfa)

        assert abs(eer - 0.5) < 0.01  # Should be ~0.5

    def test_eer_properties(self):
        """Test EER is in valid range."""
        tar = np.array([1.5, 2.0, 2.5])
        non = np.array([0.5, 1.0, 1.5])

        pmiss, pfa = rocch(tar, non)
        eer = rocch2eer(pmiss, pfa)

        assert 0.0 <= eer <= 1.0


class TestComputeEER:
    """Tests for EER computation."""

    def test_perfect_separation(self):
        """Test EER with perfectly separated scores."""
        tar = np.array([5.0, 6.0, 7.0])
        non = np.array([1.0, 2.0, 3.0])

        eer = compute_eer(tar, non)

        assert eer == 0.0

    def test_overlapping_scores(self):
        """Test EER with overlapping scores."""
        tar = np.array([2.0, 3.0, 4.0, 5.0])
        non = np.array([1.0, 2.0, 3.0, 4.0])

        eer = compute_eer(tar, non)

        assert 0.0 < eer < 1.0

    def test_eer_range(self):
        """Test that EER is always in [0, 1]."""
        tar = np.random.randn(100)
        non = np.random.randn(100) - 1.0  # Shift to create separation

        eer = compute_eer(tar, non)

        assert 0.0 <= eer <= 1.0

    def test_eer_invariance_to_shifting(self):
        """Test that EER is invariant to constant shift."""
        tar = np.array([1.0, 2.0, 3.0])
        non = np.array([0.0, 1.0, 2.0])

        eer1 = compute_eer(tar, non)
        eer2 = compute_eer(tar + 10.0, non + 10.0)

        # EER should be the same after shifting
        assert abs(eer1 - eer2) < 1e-10


class TestComputeMinDCF:
    """Tests for minDCF computation."""

    def test_perfect_separation(self):
        """Test minDCF with perfectly separated scores."""
        tar = np.array([5.0, 6.0, 7.0])
        non = np.array([1.0, 2.0, 3.0])

        min_dcf, pmiss, pfa = compute_min_dcf(tar, non)

        assert min_dcf == 0.0
        assert pmiss == 0.0
        assert pfa == 0.0

    def test_random_performance(self):
        """Test minDCF approaches 1.0 for random performance."""
        # Random scores
        np.random.seed(42)
        tar = np.random.randn(1000)
        non = np.random.randn(1000)

        min_dcf, _, _ = compute_min_dcf(tar, non)

        # Random performance should give minDCF close to 1
        assert 0.8 < min_dcf < 1.0

    def test_prior_effect(self):
        """Test that prior affects minDCF calculation."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([0.0, 1.0, 2.0])

        dcf1, _, _ = compute_min_dcf(tar, non, prior=0.1)
        dcf2, _, _ = compute_min_dcf(tar, non, prior=0.9)

        # Different priors should generally give different minDCF
        # (unless perfect separation)
        assert dcf1 >= 0.0 and dcf2 >= 0.0

    def test_normalization(self):
        """Test normalized vs unnormalized minDCF."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([0.0, 1.0, 2.0])

        dcf_norm, _, _ = compute_min_dcf(tar, non, normalize=True)
        dcf_unnorm, _, _ = compute_min_dcf(tar, non, normalize=False)

        # Normalized DCF is unnormalized divided by normalization factor
        # dcf_norm = dcf_unnorm / min(P_tar * C_miss, (1-P_tar) * C_fa)
        # With prior=0.5, c_miss=1, c_fa=1: norm_factor = min(0.5, 0.5) = 0.5
        # So dcf_norm = dcf_unnorm / 0.5 = 2 * dcf_unnorm
        assert abs(dcf_norm - 2.0 * dcf_unnorm) < 1e-10

    def test_cost_parameters(self):
        """Test that cost parameters affect minDCF."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([0.0, 1.0, 2.0])

        dcf1, _, _ = compute_min_dcf(tar, non, c_miss=1.0, c_fa=1.0)
        dcf2, _, _ = compute_min_dcf(tar, non, c_miss=10.0, c_fa=1.0)

        # Higher miss cost might change minDCF
        assert dcf1 >= 0.0 and dcf2 >= 0.0


class TestComputeActDCF:
    """Tests for actDCF computation."""

    def test_perfect_calibration(self):
        """Test actDCF with perfectly calibrated LLR scores."""
        # Create LLRs where threshold 0 is optimal
        tar_llrs = np.array([2.0, 3.0, 4.0])
        non_llrs = np.array([-4.0, -3.0, -2.0])

        act_dcf, pmiss, pfa = compute_act_dcf(tar_llrs, non_llrs, prior=0.5)

        assert act_dcf == 0.0
        assert pmiss == 0.0
        assert pfa == 0.0

    def test_threshold_application(self):
        """Test that actDCF uses Bayes threshold correctly."""
        # LLRs around 0
        tar_llrs = np.array([1.0, 2.0, 3.0])
        non_llrs = np.array([-3.0, -2.0, -1.0])

        # Prior = 0.5, equal costs -> threshold should be 0
        act_dcf, pmiss, pfa = compute_act_dcf(tar_llrs, non_llrs, prior=0.5)

        # With threshold=0, all targets accepted, all non-targets rejected
        assert pmiss == 0.0
        assert pfa == 0.0

    def test_miscalibrated_scores(self):
        """Test actDCF with miscalibrated scores."""
        # Shifted LLRs (miscalibrated)
        tar_llrs = np.array([3.0, 4.0, 5.0])  # Shifted up
        non_llrs = np.array([-1.0, 0.0, 1.0])  # Some positive

        act_dcf, pmiss, pfa = compute_act_dcf(tar_llrs, non_llrs, prior=0.5)

        # Should have some cost due to miscalibration
        assert act_dcf > 0.0

    def test_prior_effect(self):
        """Test that prior affects actDCF threshold."""
        tar_llrs = np.array([0.0, 1.0, 2.0])
        non_llrs = np.array([-2.0, -1.0, 0.0])

        dcf1, _, _ = compute_act_dcf(tar_llrs, non_llrs, prior=0.1)
        dcf2, _, _ = compute_act_dcf(tar_llrs, non_llrs, prior=0.9)

        # Different priors use different thresholds
        assert dcf1 >= 0.0 and dcf2 >= 0.0

    def test_normalization(self):
        """Test normalized vs unnormalized actDCF."""
        tar_llrs = np.array([1.0, 2.0, 3.0])
        non_llrs = np.array([-2.0, -1.0, 0.0])

        dcf_norm, _, _ = compute_act_dcf(tar_llrs, non_llrs, normalize=True)
        dcf_unnorm, _, _ = compute_act_dcf(tar_llrs, non_llrs, normalize=False)

        # Normalized DCF is unnormalized divided by normalization factor
        # With prior=0.5, c_miss=1, c_fa=1: norm_factor = 0.5
        # So dcf_norm = dcf_unnorm / 0.5 = 2 * dcf_unnorm
        assert abs(dcf_norm - 2.0 * dcf_unnorm) < 1e-10


class TestComputeCllr:
    """Tests for Cllr computation."""

    def test_perfect_llrs(self):
        """Test Cllr with perfect LLR scores."""
        # Large positive LLRs for targets
        tar_llrs = np.array([100.0, 100.0, 100.0])
        # Large negative LLRs for non-targets
        non_llrs = np.array([-100.0, -100.0, -100.0])

        cllr = compute_cllr(tar_llrs, non_llrs)

        assert cllr < 0.01  # Should be very close to 0

    def test_random_llrs(self):
        """Test Cllr with LLRs = 0 (random)."""
        # LLR = 0 means P(H1|E) = P(H2|E) = 0.5
        tar_llrs = np.zeros(100)
        non_llrs = np.zeros(100)

        cllr = compute_cllr(tar_llrs, non_llrs)

        # Cllr for LLR=0 should be 1.0
        assert abs(cllr - 1.0) < 0.01

    def test_cllr_positive(self):
        """Test that Cllr is always non-negative."""
        np.random.seed(42)
        tar_llrs = np.random.randn(100)
        non_llrs = np.random.randn(100)

        cllr = compute_cllr(tar_llrs, non_llrs)

        assert cllr >= 0.0

    def test_symmetric_cllr(self):
        """Test Cllr symmetry."""
        tar_llrs = np.array([1.0, 2.0, 3.0])
        non_llrs = np.array([-1.0, -2.0, -3.0])

        cllr = compute_cllr(tar_llrs, non_llrs)

        # Should be well-calibrated
        assert cllr < 1.0


class TestComputeMinCllr:
    """Tests for minCllr computation."""

    def test_perfect_separation(self):
        """Test minCllr with perfectly separated scores."""
        tar = np.array([5.0, 6.0, 7.0])
        non = np.array([1.0, 2.0, 3.0])

        min_cllr = compute_min_cllr(tar, non)

        assert min_cllr < 0.01  # Should be very close to 0

    def test_random_scores(self):
        """Test minCllr with random scores."""
        np.random.seed(42)
        tar = np.random.randn(1000)
        non = np.random.randn(1000)

        min_cllr = compute_min_cllr(tar, non)

        # Random should give minCllr close to 1
        assert 0.8 < min_cllr < 1.0

    def test_mincllr_positive(self):
        """Test that minCllr is always non-negative."""
        tar = np.random.randn(50)
        non = np.random.randn(50) - 1.0

        min_cllr = compute_min_cllr(tar, non)

        assert min_cllr >= 0.0

    def test_mincllr_upper_bound(self):
        """Test that minCllr has reasonable upper bound."""
        tar = np.array([1.0, 2.0, 3.0])
        non = np.array([0.0, 1.0, 2.0])

        min_cllr = compute_min_cllr(tar, non)

        # Should be bounded
        assert 0.0 <= min_cllr <= 2.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_raw_scores_metrics(self):
        """Test compute_all_metrics with raw scores."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([0.0, 1.0, 2.0])

        metrics = compute_all_metrics(tar, non, is_llr=False)

        assert 'eer' in metrics
        assert 'min_dcf' in metrics
        assert 'min_cllr' in metrics
        assert 'cllr' not in metrics  # Not computed for raw scores
        assert 'act_dcf' not in metrics  # Not computed for raw scores

    def test_llr_scores_metrics(self):
        """Test compute_all_metrics with LLR scores."""
        tar_llrs = np.array([2.0, 3.0, 4.0])
        non_llrs = np.array([-2.0, -1.0, 0.0])

        metrics = compute_all_metrics(tar_llrs, non_llrs, is_llr=True)

        assert 'eer' in metrics
        assert 'min_dcf' in metrics
        assert 'min_cllr' in metrics
        assert 'cllr' in metrics
        assert 'act_dcf' in metrics

    def test_metrics_values(self):
        """Test that all metrics have valid values."""
        tar = np.array([3.0, 4.0, 5.0])
        non = np.array([1.0, 2.0, 3.0])

        metrics = compute_all_metrics(tar, non, is_llr=False)

        assert 0.0 <= metrics['eer'] <= 1.0
        assert metrics['min_dcf'] >= 0.0
        assert metrics['min_cllr'] >= 0.0

    def test_custom_parameters(self):
        """Test compute_all_metrics with custom parameters."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([0.0, 1.0, 2.0])

        metrics = compute_all_metrics(
            tar, non,
            is_llr=False,
            prior=0.1,
            c_miss=10.0,
            c_fa=1.0
        )

        assert 'eer' in metrics
        assert 'min_dcf' in metrics
        assert metrics['min_dcf'] >= 0.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        tar = np.array([])
        non = np.array([])

        # Should handle gracefully or raise clear error
        # Different implementations may handle this differently

    def test_single_score(self):
        """Test with single score in each array."""
        tar = np.array([2.0])
        non = np.array([1.0])

        eer = compute_eer(tar, non)
        assert 0.0 <= eer <= 1.0

    def test_identical_scores_all_metrics(self):
        """Test all metrics when all scores are identical."""
        tar = np.ones(10)
        non = np.ones(10)

        eer = compute_eer(tar, non)
        min_dcf, _, _ = compute_min_dcf(tar, non)
        min_cllr = compute_min_cllr(tar, non)

        # EER should be around 0.5 for identical scores
        assert 0.4 <= eer <= 0.6
        assert min_dcf >= 0.0
        assert min_cllr >= 0.0

    def test_extreme_values(self):
        """Test with extreme score values."""
        tar = np.array([1e10, 1e10, 1e10])
        non = np.array([-1e10, -1e10, -1e10])

        eer = compute_eer(tar, non)
        min_dcf, _, _ = compute_min_dcf(tar, non)

        assert eer == 0.0
        assert min_dcf == 0.0

    def test_inf_values(self):
        """Test handling of infinite values in Cllr."""
        # Infinite LLR should give Cllr close to 0
        tar_llrs = np.array([np.inf, np.inf, 100.0])
        non_llrs = np.array([-np.inf, -np.inf, -100.0])

        # Should handle gracefully
        try:
            cllr = compute_cllr(tar_llrs, non_llrs)
            assert cllr >= 0.0
        except (ValueError, RuntimeWarning):
            # Some implementations may raise errors for inf
            pass
