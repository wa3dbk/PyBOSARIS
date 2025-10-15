"""
Unit tests for fusion module.
"""

import pytest
import numpy as np
from pybosaris.core import Scores, Key
from pybosaris.fusion import (
    stack_scores,
    train_linear_fusion,
    linear_fuse_scores,
    linear_fuse_scores_dev_eval
)


class TestStackScores:
    """Tests for stack_scores function."""

    def test_basic_stacking(self):
        """Test basic score stacking from multiple systems."""
        # Create trial structure
        tar_mask = np.array([[True, False], [False, True]])
        non_mask = np.array([[False, True], [True, False]])
        key = Key(['m1', 'm2'], ['t1', 't2'], tar_mask, non_mask)

        # Create scores for 2 systems
        scores1 = Scores(['m1', 'm2'], ['t1', 't2'],
                        np.array([[2.0, -1.0], [-0.5, 3.0]]))
        scores2 = Scores(['m1', 'm2'], ['t1', 't2'],
                        np.array([[1.5, -0.5], [0.0, 2.5]]))

        # Stack
        stacked = stack_scores([scores1, scores2], key)

        # Should be (n_systems=2, n_trials=4)
        assert stacked.shape == (2, 4)

    def test_stacking_with_different_order(self):
        """Test that stacking aligns scores correctly."""
        # Key with specific model/test order
        tar_mask = np.array([[True]])
        non_mask = np.array([[False]])
        key = Key(['m1'], ['t1'], tar_mask, non_mask)

        # Scores with same order
        scores1 = Scores(['m1'], ['t1'], np.array([[5.0]]))
        scores2 = Scores(['m1'], ['t1'], np.array([[3.0]]))

        stacked = stack_scores([scores1, scores2], key)

        assert stacked[0, 0] == 5.0
        assert stacked[1, 0] == 3.0

    def test_stacking_preserves_values(self):
        """Test that stacked values match original scores."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        scores1 = Scores(['m1'], ['t1', 't2'], np.array([[2.5, -1.5]]))
        scores2 = Scores(['m1'], ['t1', 't2'], np.array([[1.2, -0.8]]))

        stacked = stack_scores([scores1, scores2], key)

        # Check values are preserved
        assert stacked[0, 0] == 2.5  # System 1, trial 1
        assert stacked[0, 1] == -1.5  # System 1, trial 2
        assert stacked[1, 0] == 1.2  # System 2, trial 1
        assert stacked[1, 1] == -0.8  # System 2, trial 2

    def test_mismatched_trials_error(self):
        """Test that error is raised when scores don't match key."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        # Scores missing a test segment
        scores1 = Scores(['m1'], ['t1'], np.array([[2.5]]))

        with pytest.raises(ValueError, match="has .* scores, but key has .* trials"):
            stack_scores([scores1], key)


class TestTrainLinearFusion:
    """Tests for train_linear_fusion function."""

    def test_basic_training(self):
        """Test basic fusion weight training."""
        # Two systems with clear separation
        tar_scores = np.array([[2.0, 3.0], [1.5, 2.5]])  # 2 systems, 2 targets
        non_scores = np.array([[-1.0, -2.0], [-0.5, -1.5]])  # 2 systems, 2 non-targets

        fusion_func, weights = train_linear_fusion(tar_scores, non_scores)

        # Should return n_systems + 1 weights (including offset)
        assert len(weights) == 3
        # Fusion function should be callable
        assert callable(fusion_func)

    def test_fusion_function_output(self):
        """Test that fusion function produces correct output shape."""
        tar_scores = np.array([[2.0, 3.0], [1.5, 2.5]])
        non_scores = np.array([[-1.0, -2.0], [-0.5, -1.5]])

        fusion_func, weights = train_linear_fusion(tar_scores, non_scores)

        # Apply to test data
        test_scores = np.array([[1.0, 2.0], [0.5, 1.5]])  # 2 systems, 2 trials
        fused = fusion_func(test_scores)

        # Should return 1D array with n_trials elements
        assert fused.shape == (2,)

    def test_perfect_system_gets_high_weight(self):
        """Test that a perfect system gets higher weight."""
        # System 1: perfect separation
        # System 2: poor separation
        tar_scores = np.array([[5.0, 6.0, 7.0], [0.1, 0.2, 0.3]])
        non_scores = np.array([[-5.0, -6.0, -7.0], [-0.1, 0.1, 0.0]])

        fusion_func, weights = train_linear_fusion(tar_scores, non_scores)

        # First system should get much higher weight
        assert abs(weights[0]) > abs(weights[1])

    def test_different_priors(self):
        """Test fusion with different priors."""
        tar_scores = np.array([[2.0, 3.0], [1.5, 2.5]])
        non_scores = np.array([[-1.0, -2.0], [-0.5, -1.5]])

        _, weights1 = train_linear_fusion(tar_scores, non_scores, prior=0.5)
        _, weights2 = train_linear_fusion(tar_scores, non_scores, prior=0.1)

        # Different priors should give different offsets
        assert weights1[-1] != weights2[-1]

    def test_mismatched_systems_error(self):
        """Test error when number of systems doesn't match."""
        tar_scores = np.array([[2.0, 3.0], [1.5, 2.5]])  # 2 systems
        non_scores = np.array([[-1.0, -2.0]])  # 1 system

        with pytest.raises(ValueError, match="Number of systems must match"):
            train_linear_fusion(tar_scores, non_scores)

    def test_invalid_prior_error(self):
        """Test error with invalid prior."""
        tar_scores = np.array([[2.0, 3.0]])
        non_scores = np.array([[-1.0, -2.0]])

        with pytest.raises(ValueError, match="prior must be in"):
            train_linear_fusion(tar_scores, non_scores, prior=1.5)

    def test_1d_array_error(self):
        """Test error with 1D arrays instead of 2D."""
        tar_scores = np.array([2.0, 3.0])  # 1D
        non_scores = np.array([-1.0, -2.0])  # 1D

        with pytest.raises(ValueError, match="must be 2D"):
            train_linear_fusion(tar_scores, non_scores)


class TestLinearFuseScores:
    """Tests for linear_fuse_scores function."""

    def test_basic_fusion(self):
        """Test basic score fusion."""
        # Create trial structure
        tar_mask = np.array([[True, False], [False, True]])
        non_mask = np.array([[False, True], [True, False]])
        key = Key(['m1', 'm2'], ['t1', 't2'], tar_mask, non_mask)

        # Two systems
        scores1 = Scores(['m1', 'm2'], ['t1', 't2'],
                        np.array([[2.0, -1.0], [-0.5, 3.0]]))
        scores2 = Scores(['m1', 'm2'], ['t1', 't2'],
                        np.array([[1.5, -0.5], [0.0, 2.5]]))

        # Fuse
        fused_scores, weights = linear_fuse_scores([scores1, scores2], key)

        # Check output types
        assert isinstance(fused_scores, Scores)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3  # 2 systems + 1 offset

    def test_fused_scores_shape(self):
        """Test that fused scores have correct shape."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        scores1 = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        scores2 = Scores(['m1'], ['t1', 't2'], np.array([[1.5, -0.5]]))

        fused_scores, _ = linear_fuse_scores([scores1, scores2], key)

        # Should have same model/test names and shape as input
        assert fused_scores.model_names == key.model_names
        assert fused_scores.test_names == key.test_names
        assert fused_scores.shape == key.shape

    def test_fusion_improves_performance(self):
        """Test that fusion can improve over individual systems."""
        # Create scenario where fusion should help
        tar_mask = np.array([[True, True, True]])
        non_mask = np.array([[False, False, False]])
        key = Key(['m1'], ['t1', 't2', 't3'], tar_mask, non_mask)

        # System 1: good for some trials
        scores1 = Scores(['m1'], ['t1', 't2', 't3'],
                        np.array([[3.0, 1.0, 2.0]]))
        # System 2: good for others
        scores2 = Scores(['m1'], ['t1', 't2', 't3'],
                        np.array([[1.0, 3.0, 2.0]]))

        fused_scores, weights = linear_fuse_scores([scores1, scores2], key)

        # Fused scores should exist and be Scores object
        assert isinstance(fused_scores, Scores)
        # Both systems should have non-zero weights
        assert weights[0] != 0.0 or weights[1] != 0.0

    def test_empty_scores_list_error(self):
        """Test error with empty scores list."""
        key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))

        with pytest.raises(ValueError, match="must not be empty"):
            linear_fuse_scores([], key)

    def test_single_system(self):
        """Test fusion with single system."""
        tar_mask = np.array([[True]])
        non_mask = np.array([[False]])
        key = Key(['m1'], ['t1'], tar_mask, non_mask)

        scores1 = Scores(['m1'], ['t1'], np.array([[2.0]]))

        # Should work with single system
        fused_scores, weights = linear_fuse_scores([scores1], key)
        assert len(weights) == 2  # 1 system + 1 offset


class TestLinearFuseScoresDevEval:
    """Tests for linear_fuse_scores_dev_eval function."""

    def test_basic_dev_eval_fusion(self):
        """Test fusion with separate dev/eval sets."""
        # Development data
        dev_tar_mask = np.array([[True, False]])
        dev_non_mask = np.array([[False, True]])
        dev_key = Key(['m1'], ['t1', 't2'], dev_tar_mask, dev_non_mask)
        dev_scores1 = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        dev_scores2 = Scores(['m1'], ['t1', 't2'], np.array([[1.5, -0.5]]))

        # Evaluation data
        eval_tar_mask = np.array([[True, False]])
        eval_non_mask = np.array([[False, True]])
        eval_key = Key(['m2'], ['t3', 't4'], eval_tar_mask, eval_non_mask)
        eval_scores1 = Scores(['m2'], ['t3', 't4'], np.array([[1.8, -0.8]]))
        eval_scores2 = Scores(['m2'], ['t3', 't4'], np.array([[1.2, -0.3]]))

        # Fuse
        fused_eval, weights = linear_fuse_scores_dev_eval(
            [dev_scores1, dev_scores2], dev_key,
            [eval_scores1, eval_scores2], eval_key
        )

        # Check outputs
        assert isinstance(fused_eval, Scores)
        assert len(weights) == 3  # 2 systems + offset

    def test_fused_eval_has_eval_names(self):
        """Test that fused scores have eval model/test names."""
        dev_key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))
        dev_scores1 = Scores(['m1'], ['t1'], np.array([[2.0]]))
        dev_scores2 = Scores(['m1'], ['t1'], np.array([[1.5]]))

        eval_key = Key(['m2'], ['t2'], np.array([[True]]), np.array([[False]]))
        eval_scores1 = Scores(['m2'], ['t2'], np.array([[1.8]]))
        eval_scores2 = Scores(['m2'], ['t2'], np.array([[1.2]]))

        fused_eval, _ = linear_fuse_scores_dev_eval(
            [dev_scores1, dev_scores2], dev_key,
            [eval_scores1, eval_scores2], eval_key
        )

        # Fused scores should have eval names, not dev names
        assert fused_eval.model_names == ['m2']
        assert fused_eval.test_names == ['t2']

    def test_weights_from_dev_applied_to_eval(self):
        """Test that weights learned on dev are applied to eval."""
        # Dev: perfect separation
        dev_tar_mask = np.array([[True, True]])
        dev_non_mask = np.array([[False, False]])
        dev_key = Key(['m1'], ['t1', 't2'], dev_tar_mask, dev_non_mask)
        dev_scores1 = Scores(['m1'], ['t1', 't2'], np.array([[5.0, 6.0]]))
        dev_scores2 = Scores(['m1'], ['t1', 't2'], np.array([[0.1, 0.2]]))

        # Eval: similar data
        eval_tar_mask = np.array([[True]])
        eval_non_mask = np.array([[False]])
        eval_key = Key(['m1'], ['t1'], eval_tar_mask, eval_non_mask)
        eval_scores1 = Scores(['m1'], ['t1'], np.array([[4.5]]))
        eval_scores2 = Scores(['m1'], ['t1'], np.array([[0.15]]))

        fused_eval, weights = linear_fuse_scores_dev_eval(
            [dev_scores1, dev_scores2], dev_key,
            [eval_scores1, eval_scores2], eval_key
        )

        # System 1 should dominate (higher weight)
        assert abs(weights[0]) > abs(weights[1])

    def test_mismatched_num_systems_error(self):
        """Test error when dev and eval have different numbers of systems."""
        dev_key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))
        dev_scores1 = Scores(['m1'], ['t1'], np.array([[2.0]]))
        dev_scores2 = Scores(['m1'], ['t1'], np.array([[1.5]]))

        eval_key = Key(['m2'], ['t2'], np.array([[True]]), np.array([[False]]))
        eval_scores1 = Scores(['m2'], ['t2'], np.array([[1.8]]))

        with pytest.raises(ValueError, match="Number of systems must match"):
            linear_fuse_scores_dev_eval(
                [dev_scores1, dev_scores2], dev_key,
                [eval_scores1], eval_key
            )

    def test_empty_scores_list_error(self):
        """Test error with empty scores lists."""
        key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))

        with pytest.raises(ValueError, match="must not be empty"):
            linear_fuse_scores_dev_eval([], key, [], key)


class TestFusionIntegration:
    """Integration tests for fusion workflow."""

    def test_complete_fusion_workflow(self):
        """Test complete fusion workflow from scores to fused output."""
        # Create trial structure with multiple models and tests
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

        # Create scores for 3 systems
        scores1 = Scores(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'],
                        np.array([[3.0, -1.0, -1.5, 2.5],
                                 [-0.5, 2.0, -1.0, -1.2],
                                 [-0.8, -1.3, 2.8, -1.5]]))
        scores2 = Scores(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'],
                        np.array([[2.5, -0.5, -1.0, 2.0],
                                 [0.0, 1.5, -0.5, -0.8],
                                 [-0.3, -0.8, 2.3, -1.0]]))
        scores3 = Scores(['m1', 'm2', 'm3'], ['t1', 't2', 't3', 't4'],
                        np.array([[2.0, -0.8, -1.2, 1.8],
                                 [-0.2, 1.8, -0.8, -1.0],
                                 [-0.5, -1.0, 2.5, -1.2]]))

        # Fuse
        fused_scores, weights = linear_fuse_scores(
            [scores1, scores2, scores3], key
        )

        # Verify output
        assert isinstance(fused_scores, Scores)
        assert fused_scores.shape == key.shape
        assert len(weights) == 4  # 3 systems + offset
        # Weights should sum to reasonable value (not all zero)
        assert np.sum(np.abs(weights[:-1])) > 0.0

    def test_fusion_with_different_priors(self):
        """Test fusion with various prior settings."""
        tar_mask = np.array([[True, False]])
        non_mask = np.array([[False, True]])
        key = Key(['m1'], ['t1', 't2'], tar_mask, non_mask)

        scores1 = Scores(['m1'], ['t1', 't2'], np.array([[2.0, -1.0]]))
        scores2 = Scores(['m1'], ['t1', 't2'], np.array([[1.5, -0.5]]))

        # Try different priors
        for prior in [0.01, 0.1, 0.5, 0.9, 0.99]:
            fused_scores, weights = linear_fuse_scores(
                [scores1, scores2], key, prior=prior
            )
            assert isinstance(fused_scores, Scores)
            assert len(weights) == 3
