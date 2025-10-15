"""
Unit tests for Logistic Regression calibration module.
"""

import pytest
import numpy as np
from pybosaris.calibration.logistic import (
    logistic_calibration,
    logistic_calibrate_scores,
    logistic_calibrate_scores_dev_eval
)
from pybosaris.core import Scores, Key


class TestLogisticCalibration:
    """Tests for logistic_calibration function."""

    def test_basic_calibration(self):
        """Test basic logistic calibration."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([-2.0, -1.0, 0.0])

        cal_func, weights = logistic_calibration(tar, non)

        # Should return callable function
        assert callable(cal_func)
        # Weights should be [alpha, beta]
        assert len(weights) == 2
        assert isinstance(weights, np.ndarray)

    def test_weights_shape(self):
        """Test that weights have correct shape."""
        tar = np.array([1.0, 2.0, 3.0])
        non = np.array([-1.0, 0.0, 1.0])

        _, weights = logistic_calibration(tar, non)

        assert weights.shape == (2,)

    def test_calibration_is_linear(self):
        """Test that calibration is linear transformation."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([-2.0, -1.0, 0.0])

        cal_func, weights = logistic_calibration(tar, non)
        alpha, beta = weights

        # Test on various scores
        test_scores = np.array([0.0, 1.0, 2.0])
        calibrated = cal_func(test_scores)

        # Should be exactly alpha * score + beta
        expected = alpha * test_scores + beta
        np.testing.assert_array_almost_equal(calibrated, expected)

    def test_target_scores_get_higher_llr_than_nontarget(self):
        """Test that well-separated targets get higher LLR than non-targets."""
        # Clear separation
        tar = np.array([5.0, 6.0, 7.0])
        non = np.array([-5.0, -6.0, -7.0])

        cal_func, _ = logistic_calibration(tar, non)

        # Apply to representative scores
        calibrated_tar = cal_func(np.array([6.0]))
        calibrated_non = cal_func(np.array([-6.0]))

        # Target score should give higher LLR
        assert calibrated_tar[0] > calibrated_non[0]

    def test_alpha_positive_for_wellseparated_scores(self):
        """Test that alpha (scaling) is positive for well-separated scores."""
        # Targets higher than non-targets
        tar = np.array([3.0, 4.0, 5.0])
        non = np.array([-3.0, -2.0, -1.0])

        _, weights = logistic_calibration(tar, non)
        alpha = weights[0]

        # Should be positive (higher scores -> higher LLR)
        assert alpha > 0

    def test_empty_tar_error(self):
        """Test error with empty target scores."""
        tar = np.array([])
        non = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="tar_scores must not be empty"):
            logistic_calibration(tar, non)

    def test_empty_non_error(self):
        """Test error with empty non-target scores."""
        tar = np.array([1.0, 2.0])
        non = np.array([])

        with pytest.raises(ValueError, match="non_scores must not be empty"):
            logistic_calibration(tar, non)

    def test_invalid_prior_error(self):
        """Test error with invalid prior."""
        tar = np.array([1.0, 2.0])
        non = np.array([-1.0, 0.0])

        with pytest.raises(ValueError, match="prior must be between"):
            logistic_calibration(tar, non, prior=1.5)

        with pytest.raises(ValueError, match="prior must be between"):
            logistic_calibration(tar, non, prior=-0.1)

    def test_different_priors(self):
        """Test calibration with different priors."""
        tar = np.array([2.0, 3.0])
        non = np.array([0.0, 1.0])

        _, weights1 = logistic_calibration(tar, non, prior=0.1)
        _, weights2 = logistic_calibration(tar, non, prior=0.9)

        # Different priors should give different offsets (beta)
        assert abs(weights1[1] - weights2[1]) > 0.01

    def test_max_iter_parameter(self):
        """Test that max_iter parameter works."""
        tar = np.array([2.0, 3.0])
        non = np.array([0.0, 1.0])

        # Should complete with different max_iter values
        _, weights1 = logistic_calibration(tar, non, max_iter=10)
        _, weights2 = logistic_calibration(tar, non, max_iter=100)

        # Both should produce reasonable weights
        assert len(weights1) == 2
        assert len(weights2) == 2

    def test_overlapping_scores(self):
        """Test calibration with overlapping tar/non scores."""
        tar = np.array([1.0, 2.0, 3.0])
        non = np.array([1.5, 2.5, 3.5])

        cal_func, weights = logistic_calibration(tar, non)

        # Should handle without error
        test_scores = np.array([0.5, 2.0, 4.0])
        calibrated = cal_func(test_scores)
        assert len(calibrated) == 3

    def test_perfect_separation_high_alpha(self):
        """Test that perfect separation can lead to high alpha."""
        # Very clear separation
        tar = np.array([10.0, 11.0, 12.0])
        non = np.array([-10.0, -11.0, -12.0])

        _, weights = logistic_calibration(tar, non)
        alpha = weights[0]

        # Alpha should be reasonable (not infinity, but can be large)
        assert alpha > 0
        assert np.isfinite(alpha)


class TestLogisticCalibrateScores:
    """Tests for logistic_calibrate_scores function."""

    def test_basic_calibration(self):
        """Test basic score calibration."""
        # Create trial structure
        tar_mask = np.array([[True, False], [False, True]])
        non_mask = np.array([[False, True], [True, False]])
        key = Key(['m1', 'm2'], ['t1', 't2'], tar_mask, non_mask)

        # Create scores
        scores = Scores(['m1', 'm2'], ['t1', 't2'],
                       np.array([[2.0, -1.0], [-0.5, 3.0]]))

        # Calibrate
        calibrated, weights = logistic_calibrate_scores(scores, key)

        # Check output types
        assert isinstance(calibrated, Scores)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 2

    def test_calibrated_shape_matches_input(self):
        """Test that calibrated scores have same shape."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        calibrated, _ = logistic_calibrate_scores(scores, key)

        assert calibrated.model_names == scores.model_names
        assert calibrated.test_names == scores.test_names
        assert calibrated.shape == scores.shape

    def test_invalid_prior_error(self):
        """Test error with invalid prior."""
        key = Key(['m1'], ['t1', 't2'], np.array([[True, False]]), np.array([[False, True]]))
        scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))

        with pytest.raises(ValueError, match="prior must be between"):
            logistic_calibrate_scores(scores, key, prior=2.0)

    def test_no_targets_error(self):
        """Test error when no target trials."""
        # All non-targets
        tar_mask = np.array([[False, False]])
        non_mask = np.array([[True, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)
        scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))

        with pytest.raises(ValueError, match="No target scores"):
            logistic_calibrate_scores(scores, key)

    def test_no_nontargets_error(self):
        """Test error when no non-target trials."""
        # All targets
        tar_mask = np.array([[True, True]])
        non_mask = np.array([[False, False]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)
        scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, 3.0]]))

        with pytest.raises(ValueError, match="No non-target scores"):
            logistic_calibrate_scores(scores, key)

    def test_calibration_with_multiple_trials(self):
        """Test calibration with multiple trials."""
        # Create larger trial structure
        tar_mask = np.array([
            [True, False, False, True],
            [False, True, False, False]
        ])
        non_mask = np.array([
            [False, True, True, False],
            [True, False, True, True]
        ])
        key = Key(['m1', 'm2'], ['t1', 't2', 't3', 't4'], tar_mask, non_mask)

        scores = Scores(['m1', 'm2'], ['t1', 't2', 't3', 't4'],
                       np.array([[3.0, -1.0, -1.5, 2.5],
                                [-0.5, 2.0, -1.0, -1.2]]))

        calibrated, weights = logistic_calibrate_scores(scores, key)

        assert isinstance(calibrated, Scores)
        assert calibrated.shape == scores.shape
        assert len(weights) == 2

    def test_calibration_is_linear_transform(self):
        """Test that calibration applies linear transformation."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        calibrated, weights = logistic_calibrate_scores(scores, key)

        alpha, beta = weights

        # Manually compute expected calibrated scores
        expected_mat = alpha * scores.score_mat + beta
        np.testing.assert_array_almost_equal(
            calibrated.score_mat, expected_mat, decimal=5
        )


class TestLogisticCalibrateScoresDevEval:
    """Tests for logistic_calibrate_scores_dev_eval function."""

    def test_basic_dev_eval_calibration(self):
        """Test calibration with dev/eval split."""
        # Development data
        dev_tar_mask = np.array([[True, False]])
        dev_non_mask = np.array([[False, True]])
        dev_key = Key(['m1'], ['t1', 't2'], dev_tar_mask, dev_non_mask)
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))

        # Evaluation data
        eval_tar_mask = np.array([[True, False]])
        eval_non_mask = np.array([[False, True]])
        eval_key = Key(['m2'], ['t3', 't4'], eval_tar_mask, eval_non_mask)
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        # Calibrate
        calibrated_eval, weights = logistic_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        assert isinstance(calibrated_eval, Scores)
        assert len(weights) == 2

    def test_eval_retains_eval_names(self):
        """Test that calibrated eval scores have eval model/test names."""
        dev_key = Key(['m1'], ['t1', 't2'], np.array([[True, False]]), np.array([[False, True]]))
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))

        eval_key = Key(['m2'], ['t3', 't4'], np.array([[True, False]]), np.array([[False, True]]))
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        calibrated, _ = logistic_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        assert calibrated.model_names == eval_scores.model_names
        assert calibrated.test_names == eval_scores.test_names

    def test_invalid_prior_error(self):
        """Test error with invalid prior."""
        dev_key = Key(['m1'], ['t1', 't2'], np.array([[True, False]]), np.array([[False, True]]))
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        eval_key = Key(['m2'], ['t3', 't4'], np.array([[True, False]]), np.array([[False, True]]))
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        with pytest.raises(ValueError, match="prior must be between"):
            logistic_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key, prior=1.1
            )

    def test_no_dev_targets_error(self):
        """Test error when dev has no targets."""
        dev_key = Key(['m1'], ['t1', 't2'], np.array([[False, False]]), np.array([[True, True]]))
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        eval_key = Key(['m2'], ['t3', 't4'], np.array([[True, False]]), np.array([[False, True]]))
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        with pytest.raises(ValueError, match="No target scores in development"):
            logistic_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key
            )

    def test_no_dev_nontargets_error(self):
        """Test error when dev has no non-targets."""
        dev_key = Key(['m1'], ['t1', 't2'], np.array([[True, True]]), np.array([[False, False]]))
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, 3.0]]))
        eval_key = Key(['m2'], ['t3', 't4'], np.array([[True, False]]), np.array([[False, True]]))
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        with pytest.raises(ValueError, match="No non-target scores in development"):
            logistic_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key
            )

    def test_weights_from_dev_applied_to_eval(self):
        """Test that weights learned on dev are applied to eval."""
        # Dev: clear separation
        dev_tar_mask = np.array([[True, True, False, False]])
        dev_non_mask = np.array([[False, False, True, True]])
        dev_key = Key(['m1'], ['t1', 't2', 't3', 't4'],
                     dev_tar_mask, dev_non_mask)
        dev_scores = Scores(['m1'], ['t1', 't2', 't3', 't4'],
                           np.array([[3.0, 3.5, -2.0, -2.5]]))

        # Eval: similar data
        eval_tar_mask = np.array([[True, False]])
        eval_non_mask = np.array([[False, True]])
        eval_key = Key(['m1'], ['t5', 't6'], eval_tar_mask, eval_non_mask)
        eval_scores = Scores(['m1'], ['t5', 't6'],
                            np.array([[3.2, -2.2]]))

        calibrated, weights = logistic_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        alpha, beta = weights

        # Verify weights are applied: calibrated = alpha * eval + beta
        expected_mat = alpha * eval_scores.score_mat + beta
        np.testing.assert_array_almost_equal(
            calibrated.score_mat, expected_mat, decimal=5
        )

    def test_calibration_with_different_priors(self):
        """Test dev/eval calibration with various priors."""
        dev_key = Key(['m1'], ['t1', 't2'], np.array([[True, False]]), np.array([[False, True]]))
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        eval_key = Key(['m2'], ['t3', 't4'], np.array([[True, False]]), np.array([[False, True]]))
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        for prior in [0.01, 0.1, 0.5, 0.9, 0.99]:
            calibrated, weights = logistic_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key, prior=prior
            )
            assert isinstance(calibrated, Scores)
            assert len(weights) == 2


class TestLogisticIntegration:
    """Integration tests for logistic calibration workflow."""

    def test_complete_calibration_workflow(self):
        """Test complete calibration from raw scores to calibrated LLRs."""
        # Create trial structure
        tar_mask = np.array([
            [True, False, False, True],
            [False, True, False, False],
            [False, False, True, False]
        ])
        non_mask = np.array([
            [False, True, True, False],
            [True, False, True, True],
            [True, True, False, True]
        ])
        key = Key(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'], tar_mask, non_mask)

        # Raw scores with clear tar/non separation
        scores = Scores(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'],
                       np.array([[3.0, -2.0, -1.5, 2.5],
                                [-0.5, 2.5, -1.0, -1.2],
                                [-0.8, -1.3, 2.8, -1.5]]))

        # Calibrate
        calibrated, weights = logistic_calibrate_scores(scores, key)

        # Weights should be reasonable
        alpha, beta = weights
        assert alpha > 0  # Positive scaling
        assert np.isfinite(alpha) and np.isfinite(beta)

        # Verify calibration is linear transform
        expected_mat = alpha * scores.score_mat + beta
        np.testing.assert_array_almost_equal(
            calibrated.score_mat, expected_mat, decimal=5
        )

    def test_dev_eval_generalization(self):
        """Test that dev-trained calibration generalizes to eval."""
        # Dev: well-separated scores
        dev_tar_mask = np.array([[True, True, False, False]])
        dev_non_mask = np.array([[False, False, True, True]])
        dev_key = Key(['m1'], ['t1', 't2', 't3', 't4'],
                     dev_tar_mask, dev_non_mask)
        dev_scores = Scores(['m1'], ['t1', 't2', 't3', 't4'],
                           np.array([[3.0, 3.5, -2.0, -2.5]]))

        # Eval: similar score ranges
        eval_tar_mask = np.array([[True, False]])
        eval_non_mask = np.array([[False, True]])
        eval_key = Key(['m1'], ['t5', 't6'], eval_tar_mask, eval_non_mask)
        eval_scores = Scores(['m1'], ['t5', 't6'],
                            np.array([[3.2, -2.2]]))

        # Calibrate
        calibrated, weights = logistic_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        # Get calibrated scores
        cal_tar, cal_non = calibrated.get_tar_non(eval_key)

        # With good separation, target should be higher than non-target
        assert cal_tar[0] > cal_non[0]

    def test_comparison_with_pav(self):
        """Test that logistic calibration produces different results than PAV."""
        from pybosaris.calibration.pav import pav_calibrate_scores

        tar_mask = np.array([[True, False, False, True]])
        non_mask = np.array([[False, True, True, False]])
        key = Key(['m1'], ['t1', 't2', 't3', 't4'], tar_mask, non_mask)
        scores = Scores(['m1'], ['t1', 't2', 't3', 't4'],
                       np.array([[3.0, -1.0, -1.5, 2.5]]))

        # Calibrate with both methods
        log_calibrated, _ = logistic_calibrate_scores(scores, key)
        pav_calibrated = pav_calibrate_scores(scores, key)

        # Results should be different (PAV is non-parametric, LR is linear)
        # They might be similar but shouldn't be identical
        assert not np.allclose(log_calibrated.score_mat, pav_calibrated.score_mat, rtol=1e-3)
