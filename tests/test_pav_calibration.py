"""
Unit tests for PAV (Pool Adjacent Violators) calibration module.
"""

import pytest
import numpy as np
from pybosaris.calibration.pav import (
    pavx,
    pav_calibration,
    pav_calibrate_scores,
    pav_calibrate_scores_dev_eval,
    _make_boundary_indices,
    _pav_transform_impl
)
from pybosaris.core import Scores, Key


class TestPavx:
    """Tests for pavx algorithm."""

    def test_monotonic_input(self):
        """Test pavx with already monotonic input."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ghat, width, height = pavx(y)

        # Should return identical to input
        np.testing.assert_array_almost_equal(ghat, y)
        # Should have n bins (no pooling needed)
        assert len(width) == 5
        assert len(height) == 5

    def test_single_violation(self):
        """Test pavx with single violation."""
        y = np.array([1.0, 3.0, 2.0, 4.0])
        ghat, width, height = pavx(y)

        # Output must be monotonic
        assert np.all(np.diff(ghat) >= -1e-10)
        # Middle values should be pooled to 2.5
        assert abs(ghat[1] - 2.5) < 1e-10
        assert abs(ghat[2] - 2.5) < 1e-10

    def test_multiple_violations(self):
        """Test pavx with multiple violations."""
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ghat, width, height = pavx(y)

        # Should pool everything to mean
        expected = np.full(5, 3.0)
        np.testing.assert_array_almost_equal(ghat, expected)
        assert len(width) == 1
        assert width[0] == 5

    def test_output_monotonicity(self):
        """Test that output is always monotonic."""
        np.random.seed(42)
        for _ in range(10):
            y = np.random.randn(20)
            ghat, width, height = pavx(y)
            # Check monotonicity
            assert np.all(np.diff(ghat) >= -1e-10)

    def test_bin_properties(self):
        """Test properties of width and height."""
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        ghat, width, height = pavx(y)

        # Sum of widths should equal length of input
        assert sum(width) == len(y)
        # Heights should be monotonically increasing
        assert np.all(np.diff(height) >= -1e-10)

    def test_empty_input_error(self):
        """Test error with empty input."""
        y = np.array([])
        with pytest.raises(ValueError, match="must not be empty"):
            pavx(y)

    def test_single_element(self):
        """Test with single element."""
        y = np.array([42.0])
        ghat, width, height = pavx(y)

        np.testing.assert_array_almost_equal(ghat, y)
        assert len(width) == 1
        assert width[0] == 1
        assert height[0] == 42.0

    def test_two_elements_increasing(self):
        """Test with two increasing elements."""
        y = np.array([1.0, 2.0])
        ghat, width, height = pavx(y)

        np.testing.assert_array_almost_equal(ghat, y)
        assert len(width) == 2

    def test_two_elements_decreasing(self):
        """Test with two decreasing elements."""
        y = np.array([2.0, 1.0])
        ghat, width, height = pavx(y)

        # Should pool to mean
        expected = np.array([1.5, 1.5])
        np.testing.assert_array_almost_equal(ghat, expected)
        assert len(width) == 1

    def test_constant_input(self):
        """Test with constant values."""
        y = np.ones(10) * 3.14
        ghat, width, height = pavx(y)

        np.testing.assert_array_almost_equal(ghat, y)
        assert len(width) == 1
        assert width[0] == 10


class TestPavCalibration:
    """Tests for pav_calibration function."""

    def test_basic_calibration(self):
        """Test basic PAV calibration."""
        tar = np.array([2.0, 3.0, 4.0])
        non = np.array([-2.0, -1.0, 0.0])

        pav_trans, score_bounds, llr_bounds = pav_calibration(tar, non)

        # Should return callable function
        assert callable(pav_trans)
        # Bounds should be arrays
        assert isinstance(score_bounds, np.ndarray)
        assert isinstance(llr_bounds, np.ndarray)
        # Bounds should have same length
        assert len(score_bounds) == len(llr_bounds)

    def test_transformation_monotonicity(self):
        """Test that transformation is monotonically increasing."""
        tar = np.array([2.0, 3.0, 4.0, 5.0])
        non = np.array([-2.0, -1.0, 0.0, 1.0])

        pav_trans, _, _ = pav_calibration(tar, non)

        # Test on range of scores
        test_scores = np.linspace(-3, 6, 100)
        calibrated = pav_trans(test_scores)

        # Should be monotonically increasing
        assert np.all(np.diff(calibrated) >= -1e-6)

    def test_target_scores_get_positive_llr(self):
        """Test that well-separated target scores get positive LLR."""
        tar = np.array([5.0, 6.0, 7.0])
        non = np.array([-5.0, -6.0, -7.0])

        pav_trans, _, _ = pav_calibration(tar, non)

        # Apply to target scores
        calibrated_tar = pav_trans(tar)

        # Should be positive (favoring target hypothesis)
        assert np.all(calibrated_tar > 0)

    def test_nontarget_scores_get_negative_llr(self):
        """Test that well-separated non-target scores get negative LLR."""
        tar = np.array([5.0, 6.0, 7.0])
        non = np.array([-5.0, -6.0, -7.0])

        pav_trans, _, _ = pav_calibration(tar, non)

        # Apply to non-target scores
        calibrated_non = pav_trans(non)

        # Should be negative (favoring non-target hypothesis)
        assert np.all(calibrated_non < 0)

    def test_extrapolation_beyond_training(self):
        """Test transformation extrapolates beyond training data."""
        tar = np.array([2.0, 3.0])
        non = np.array([0.0, 1.0])

        pav_trans, _, _ = pav_calibration(tar, non)

        # Test scores outside training range
        extreme_scores = np.array([-100.0, 100.0])
        calibrated = pav_trans(extreme_scores)

        # Should handle without error
        assert len(calibrated) == 2
        # Very low score should give very negative LLR
        assert calibrated[0] < 0
        # Very high score should give very positive LLR
        assert calibrated[1] > 0

    def test_empty_tar_error(self):
        """Test error with empty target scores."""
        tar = np.array([])
        non = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="tar_scores must not be empty"):
            pav_calibration(tar, non)

    def test_empty_non_error(self):
        """Test error with empty non-target scores."""
        tar = np.array([1.0, 2.0])
        non = np.array([])

        with pytest.raises(ValueError, match="non_scores must not be empty"):
            pav_calibration(tar, non)

    def test_custom_score_offset(self):
        """Test with custom score offset."""
        tar = np.array([2.0, 3.0])
        non = np.array([0.0, 1.0])

        _, _, llr1 = pav_calibration(tar, non, score_offset=1e-6)
        _, _, llr2 = pav_calibration(tar, non, score_offset=1e-3)

        # Different offsets should give different LLR bounds
        # (though may be very similar)
        assert llr1.shape == llr2.shape

    def test_overlapping_scores(self):
        """Test calibration with overlapping tar/non scores."""
        tar = np.array([1.0, 2.0, 3.0])
        non = np.array([1.5, 2.5, 3.5])

        pav_trans, _, _ = pav_calibration(tar, non)

        # Should handle without error
        test_scores = np.array([0.5, 2.0, 4.0])
        calibrated = pav_trans(test_scores)
        assert len(calibrated) == 3


class TestMakeBoundaryIndices:
    """Tests for _make_boundary_indices helper."""

    def test_single_bin(self):
        """Test with single bin."""
        width = np.array([5])
        indices = _make_boundary_indices(width)

        # Should have 2 indices (left and right)
        assert len(indices) == 2
        assert indices[0] == 0  # Left
        assert indices[1] == 4  # Right (0-indexed)

    def test_multiple_bins(self):
        """Test with multiple bins."""
        width = np.array([2, 3, 1])
        indices = _make_boundary_indices(width)

        # Should have 6 indices (2 per bin)
        assert len(indices) == 6
        # Bin 1: [0, 1]
        assert indices[0] == 0
        assert indices[1] == 1
        # Bin 2: [2, 4]
        assert indices[2] == 2
        assert indices[3] == 4
        # Bin 3: [5, 5]
        assert indices[4] == 5
        assert indices[5] == 5

    def test_interleaved_format(self):
        """Test that indices are interleaved (left, right, left, right, ...)."""
        width = np.array([1, 1, 1])
        indices = _make_boundary_indices(width)

        # Should alternate left/right
        assert indices[0] == 0  # Left of bin 1
        assert indices[1] == 0  # Right of bin 1
        assert indices[2] == 1  # Left of bin 2
        assert indices[3] == 1  # Right of bin 2
        assert indices[4] == 2  # Left of bin 3
        assert indices[5] == 2  # Right of bin 3


class TestPavTransformImpl:
    """Tests for _pav_transform_impl helper."""

    def test_basic_transformation(self):
        """Test basic piecewise linear transformation."""
        # Simple 2-segment function
        score_bounds = np.array([0.0, 1.0, 2.0])
        llr_bounds = np.array([0.0, 1.0, 2.0])

        scores = np.array([0.5, 1.5])
        result = _pav_transform_impl(scores, score_bounds, llr_bounds)

        np.testing.assert_array_almost_equal(result, [0.5, 1.5])

    def test_interpolation(self):
        """Test linear interpolation between bounds."""
        score_bounds = np.array([0.0, 10.0])
        llr_bounds = np.array([0.0, 100.0])

        scores = np.array([5.0])  # Midpoint
        result = _pav_transform_impl(scores, score_bounds, llr_bounds)

        assert abs(result[0] - 50.0) < 1e-10

    def test_extrapolation_low(self):
        """Test behavior with scores below range."""
        score_bounds = np.array([0.0, 1.0])
        llr_bounds = np.array([0.0, 1.0])

        scores = np.array([-5.0])
        result = _pav_transform_impl(scores, score_bounds, llr_bounds)

        # Should use first segment
        assert len(result) == 1

    def test_extrapolation_high(self):
        """Test behavior with scores above range."""
        score_bounds = np.array([0.0, 1.0])
        llr_bounds = np.array([0.0, 1.0])

        scores = np.array([10.0])
        result = _pav_transform_impl(scores, score_bounds, llr_bounds)

        # Should use last segment
        assert len(result) == 1

    def test_shape_preservation(self):
        """Test that input shape is preserved."""
        score_bounds = np.array([0.0, 1.0])
        llr_bounds = np.array([0.0, 1.0])

        # 2D input
        scores = np.array([[0.2, 0.5], [0.7, 0.9]])
        result = _pav_transform_impl(scores, score_bounds, llr_bounds)

        assert result.shape == scores.shape


class TestPavCalibrateScores:
    """Tests for pav_calibrate_scores function."""

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
        calibrated = pav_calibrate_scores(scores, key)

        # Check output type
        assert isinstance(calibrated, Scores)
        assert calibrated.shape == scores.shape

    def test_calibrated_shape_matches_input(self):
        """Test that calibrated scores have same shape."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        calibrated = pav_calibrate_scores(scores, key)

        assert calibrated.model_names == scores.model_names
        assert calibrated.test_names == scores.test_names

    def test_invalid_prior_error(self):
        """Test error with invalid prior."""
        key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))
        scores = Scores(['m1'], ['t1'], np.array([[2.0]]))

        with pytest.raises(ValueError, match="prior must be between"):
            pav_calibrate_scores(scores, key, prior=1.5)

    def test_no_targets_error(self):
        """Test error when no target trials."""
        # All non-targets
        tar_mask = np.array([[False]])
        non_mask = np.array([[True]])
        key = Key(['m1'], ['t1'], tar_mask, non_mask)
        scores = Scores(['m1'], ['t1'], np.array([[2.0]]))

        with pytest.raises(ValueError, match="No target scores"):
            pav_calibrate_scores(scores, key)

    def test_no_nontargets_error(self):
        """Test error when no non-target trials."""
        # All targets
        tar_mask = np.array([[True]])
        non_mask = np.array([[False]])
        key = Key(['m1'], ['t1'], tar_mask, non_mask)
        scores = Scores(['m1'], ['t1'], np.array([[2.0]]))

        with pytest.raises(ValueError, match="No non-target scores"):
            pav_calibrate_scores(scores, key)

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

        calibrated = pav_calibrate_scores(scores, key)

        assert isinstance(calibrated, Scores)
        assert calibrated.shape == scores.shape


class TestPavCalibrateScoresDevEval:
    """Tests for pav_calibrate_scores_dev_eval function."""

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
        calibrated_eval = pav_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        assert isinstance(calibrated_eval, Scores)

    def test_eval_retains_eval_names(self):
        """Test that calibrated eval scores have eval model/test names."""
        dev_key = Key(['m1'], ['t1', 't2'], np.array([[True, False]]), np.array([[False, True]]))
        dev_scores = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))

        eval_key = Key(['m2'], ['t3', 't4'], np.array([[True, False]]), np.array([[False, True]]))
        eval_scores = Scores(['m2'], ['t3', 't4'], np.array([[1.5, -0.5]]))

        calibrated = pav_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        assert calibrated.model_names == eval_scores.model_names
        assert calibrated.test_names == eval_scores.test_names

    def test_invalid_prior_error(self):
        """Test error with invalid prior."""
        dev_key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))
        dev_scores = Scores(['m1'], ['t1'], np.array([[2.0]]))
        eval_key = Key(['m2'], ['t2'], np.array([[True]]), np.array([[False]]))
        eval_scores = Scores(['m2'], ['t2'], np.array([[1.5]]))

        with pytest.raises(ValueError, match="prior must be between"):
            pav_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key, prior=-0.1
            )

    def test_no_dev_targets_error(self):
        """Test error when dev has no targets."""
        dev_key = Key(['m1'], ['t1'], np.array([[False]]), np.array([[True]]))
        dev_scores = Scores(['m1'], ['t1'], np.array([[2.0]]))
        eval_key = Key(['m2'], ['t2'], np.array([[True]]), np.array([[False]]))
        eval_scores = Scores(['m2'], ['t2'], np.array([[1.5]]))

        with pytest.raises(ValueError, match="No target scores in development"):
            pav_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key
            )

    def test_no_dev_nontargets_error(self):
        """Test error when dev has no non-targets."""
        dev_key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))
        dev_scores = Scores(['m1'], ['t1'], np.array([[2.0]]))
        eval_key = Key(['m2'], ['t2'], np.array([[True]]), np.array([[False]]))
        eval_scores = Scores(['m2'], ['t2'], np.array([[1.5]]))

        with pytest.raises(ValueError, match="No non-target scores in development"):
            pav_calibrate_scores_dev_eval(
                dev_scores, dev_key, eval_scores, eval_key
            )


class TestPavIntegration:
    """Integration tests for PAV calibration workflow."""

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
        calibrated = pav_calibrate_scores(scores, key)

        # Get calibrated tar/non scores
        cal_tar, cal_non = calibrated.get_tar_non(key)

        # Most targets should have positive LLR
        assert np.mean(cal_tar > 0) > 0.5
        # Most non-targets should have negative LLR
        assert np.mean(cal_non < 0) > 0.5

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
        calibrated = pav_calibrate_scores_dev_eval(
            dev_scores, dev_key, eval_scores, eval_key
        )

        # Get calibrated scores
        cal_tar, cal_non = calibrated.get_tar_non(eval_key)

        # Target should be positive, non-target negative
        assert cal_tar[0] > 0
        assert cal_non[0] < 0
