"""
Unit tests for mathematical utility functions.
"""

import pytest
import numpy as np
from pybosaris.utils import (
    logit,
    sigmoid,
    probit,
    inv_probit,
    effective_prior,
    neg_log_sigmoid,
    safe_log
)


class TestLogit:
    """Tests for logit function."""

    def test_logit_basic(self):
        """Test basic logit functionality."""
        assert np.isclose(logit(0.5), 0.0)
        assert logit(0.75) > 0
        assert logit(0.25) < 0

    def test_logit_bounds(self):
        """Test logit at boundaries."""
        assert logit(0.0) == -np.inf
        assert logit(1.0) == np.inf

    def test_logit_array(self):
        """Test logit with arrays."""
        p = np.array([0.1, 0.5, 0.9])
        result = logit(p)
        assert len(result) == 3
        assert result[1] == 0.0  # logit(0.5) = 0

    def test_logit_inverse_sigmoid(self):
        """Test that logit is inverse of sigmoid."""
        p = 0.7
        assert np.isclose(sigmoid(logit(p)), p)


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_basic(self):
        """Test basic sigmoid functionality."""
        assert np.isclose(sigmoid(0.0), 0.5)
        assert sigmoid(10.0) > 0.9
        assert sigmoid(-10.0) < 0.1

    def test_sigmoid_numerical_stability(self):
        """Test sigmoid numerical stability for large values."""
        # Should not overflow
        result = sigmoid(np.array([-1000, 0, 1000]))
        assert all(np.isfinite(result))
        assert result[0] < 1e-10
        assert np.isclose(result[1], 0.5)
        assert result[2] > 1 - 1e-10

    def test_sigmoid_inverse_logit(self):
        """Test that sigmoid is inverse of logit."""
        x = 2.5
        assert np.isclose(logit(sigmoid(x)), x)


class TestProbit:
    """Tests for probit function."""

    def test_probit_basic(self):
        """Test basic probit functionality."""
        assert np.isclose(probit(0.5), 0.0)
        assert probit(0.75) > 0
        assert probit(0.25) < 0

    def test_probit_inverse(self):
        """Test that probit and inv_probit are inverses."""
        p = 0.7
        assert np.isclose(inv_probit(probit(p)), p)

        x = 1.5
        assert np.isclose(probit(inv_probit(x)), x)


class TestEffectivePrior:
    """Tests for effective_prior function."""

    def test_effective_prior_equal_costs(self):
        """Test effective prior with equal costs."""
        eff_prior = effective_prior(0.5, cost_miss=1.0, cost_fa=1.0)
        assert np.isclose(eff_prior, 0.5)

    def test_effective_prior_different_costs(self):
        """Test effective prior with different costs."""
        # Higher miss cost should increase effective prior
        eff_prior = effective_prior(0.5, cost_miss=10.0, cost_fa=1.0)
        assert eff_prior > 0.5

        # Higher FA cost should decrease effective prior
        eff_prior = effective_prior(0.5, cost_miss=1.0, cost_fa=10.0)
        assert eff_prior < 0.5

    def test_effective_prior_bounds(self):
        """Test effective prior is always between 0 and 1."""
        eff_prior = effective_prior(0.1, cost_miss=100.0, cost_fa=1.0)
        assert 0 < eff_prior < 1

    def test_effective_prior_validation(self):
        """Test effective prior validates inputs."""
        with pytest.raises(ValueError):
            effective_prior(-0.1)  # Invalid prior
        with pytest.raises(ValueError):
            effective_prior(1.5)  # Invalid prior
        with pytest.raises(ValueError):
            effective_prior(0.5, cost_miss=-1.0)  # Negative cost


class TestNegLogSigmoid:
    """Tests for neg_log_sigmoid function."""

    def test_neg_log_sigmoid_basic(self):
        """Test basic functionality."""
        result = neg_log_sigmoid(0.0)
        assert np.isclose(result, np.log(2))

    def test_neg_log_sigmoid_numerical_stability(self):
        """Test numerical stability for large values."""
        # Should not overflow or underflow
        result = neg_log_sigmoid(np.array([-100, 0, 100]))
        assert all(np.isfinite(result))

    def test_neg_log_sigmoid_equivalence(self):
        """Test equivalence with -log(sigmoid(x))."""
        x = np.array([-5, -1, 0, 1, 5])
        result1 = neg_log_sigmoid(x)
        result2 = -np.log(sigmoid(x))
        assert np.allclose(result1, result2)


class TestSafeLog:
    """Tests for safe_log function."""

    def test_safe_log_basic(self):
        """Test basic functionality."""
        assert np.isclose(safe_log(np.e), 1.0)
        assert np.isclose(safe_log(1.0), 0.0)

    def test_safe_log_clipping(self):
        """Test that zero is clipped to epsilon."""
        result = safe_log(0.0, eps=1e-10)
        assert np.isfinite(result)
        assert result < 0  # log(eps) is negative

    def test_safe_log_negative_values(self):
        """Test handling of negative values."""
        result = safe_log(-1.0, eps=1e-10)
        # Should clip to eps
        assert np.isfinite(result)

    def test_safe_log_array(self):
        """Test with arrays."""
        x = np.array([0.0, 1.0, 10.0, 0.0])
        result = safe_log(x)
        assert all(np.isfinite(result))
