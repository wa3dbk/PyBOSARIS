"""
Unit tests for Ndx class.
"""

import pytest
import numpy as np
from pybosaris.core import Ndx


class TestNdxConstructor:
    """Tests for Ndx constructor."""

    def test_empty_constructor(self):
        """Test empty Ndx constructor."""
        ndx = Ndx()
        assert ndx.n_models == 0
        assert ndx.n_tests == 0
        assert ndx.n_trials == 0

    def test_basic_constructor(self):
        """Test basic Ndx constructor."""
        ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'])
        assert ndx.n_models == 2
        assert ndx.n_tests == 3
        assert ndx.n_trials == 6  # All combinations by default
        assert ndx.trial_mask.all()

    def test_constructor_with_mask(self):
        """Test constructor with custom trial mask."""
        mask = np.array([[True, False, True], [False, True, False]])
        ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'], mask)
        assert ndx.n_trials == 3

    def test_constructor_invalid_mask_shape(self):
        """Test that invalid mask shape raises error."""
        mask = np.array([[True, False]])  # Wrong shape
        with pytest.raises(ValueError):
            Ndx(['m1', 'm2'], ['t1', 't2', 't3'], mask)


class TestNdxProperties:
    """Tests for Ndx properties."""

    def test_shape_property(self):
        """Test shape property."""
        ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'])
        assert ndx.shape == (2, 3)

    def test_n_trials_property(self):
        """Test n_trials property."""
        mask = np.array([[True, False], [True, True]])
        ndx = Ndx(['m1', 'm2'], ['t1', 't2'], mask)
        assert ndx.n_trials == 3


class TestNdxMethods:
    """Tests for Ndx methods."""

    def test_repr(self):
        """Test string representation."""
        ndx = Ndx(['m1'], ['t1', 't2'])
        repr_str = repr(ndx)
        assert 'Ndx' in repr_str
        assert '1 models' in repr_str
        assert '2 tests' in repr_str

    def test_equality(self):
        """Test equality comparison."""
        ndx1 = Ndx(['m1', 'm2'], ['t1', 't2'])
        ndx2 = Ndx(['m1', 'm2'], ['t1', 't2'])
        ndx3 = Ndx(['m1'], ['t1', 't2'])

        assert ndx1 == ndx2
        assert ndx1 != ndx3
        assert ndx1 != "not an ndx"  # Test with non-Ndx type
        assert ndx1 != 123  # Test with another non-Ndx type

    def test_validate(self):
        """Test validation."""
        ndx = Ndx(['m1', 'm2'], ['t1', 't2'])
        assert ndx.validate()

    def test_copy(self):
        """Test deep copy."""
        ndx1 = Ndx(['m1', 'm2'], ['t1', 't2'])
        ndx2 = ndx1.copy()

        assert ndx1 == ndx2
        assert ndx1 is not ndx2
        assert ndx1.trial_mask is not ndx2.trial_mask

    def test_to_dict_from_dict(self):
        """Test serialization to/from dict."""
        ndx1 = Ndx(['m1', 'm2'], ['t1', 't2'])
        d = ndx1.to_dict()

        assert 'model_names' in d
        assert 'test_names' in d
        assert 'trial_mask' in d

        ndx2 = Ndx.from_dict(d)
        assert ndx1 == ndx2

    def test_save_requires_h5py(self):
        """Test that save raises ImportError when h5py not available."""
        ndx = Ndx(['m1'], ['t1'])
        with pytest.raises(ImportError, match="h5py is required"):
            ndx.save('test.h5')

    def test_load_requires_h5py(self):
        """Test that load raises ImportError when h5py not available."""
        with pytest.raises(ImportError, match="h5py is required"):
            Ndx.load('test.h5')


class TestNdxFilter:
    """Tests for Ndx filter method."""

    def test_filter_models_keep(self):
        """Test filtering models (keep mode)."""
        ndx = Ndx(['m1', 'm2', 'm3'], ['t1', 't2'])
        filtered = ndx.filter(model_list=['m1', 'm2'], keep=True)

        assert filtered.n_models == 2
        assert filtered.model_names == ['m1', 'm2']
        assert filtered.n_tests == 2

    def test_filter_models_remove(self):
        """Test filtering models (remove mode)."""
        ndx = Ndx(['m1', 'm2', 'm3'], ['t1', 't2'])
        filtered = ndx.filter(model_list=['m3'], keep=False)

        assert filtered.n_models == 2
        assert filtered.model_names == ['m1', 'm2']

    def test_filter_tests_keep(self):
        """Test filtering test segments (keep mode)."""
        ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'])
        filtered = ndx.filter(test_list=['t1', 't3'], keep=True)

        assert filtered.n_tests == 2
        assert filtered.test_names == ['t1', 't3']

    def test_filter_both(self):
        """Test filtering both models and tests."""
        ndx = Ndx(['m1', 'm2', 'm3'], ['t1', 't2', 't3'])
        filtered = ndx.filter(
            model_list=['m1', 'm2'],
            test_list=['t2', 't3'],
            keep=True
        )

        assert filtered.n_models == 2
        assert filtered.n_tests == 2

    def test_filter_tests_remove(self):
        """Test filtering tests (remove mode)."""
        ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'])
        filtered = ndx.filter(test_list=['t3'], keep=False)

        assert filtered.n_tests == 2
        assert filtered.test_names == ['t1', 't2']


class TestNdxMerge:
    """Tests for Ndx merge method."""

    def test_merge_empty_list(self):
        """Test merging empty list."""
        merged = Ndx.merge([])
        assert merged.n_models == 0
        assert merged.n_tests == 0

    def test_merge_single(self):
        """Test merging single Ndx."""
        ndx = Ndx(['m1'], ['t1', 't2'])
        merged = Ndx.merge([ndx])
        assert merged == ndx

    def test_merge_two(self):
        """Test merging two Ndx objects."""
        ndx1 = Ndx(['m1'], ['t1', 't2'])
        ndx2 = Ndx(['m2'], ['t2', 't3'])

        merged = Ndx.merge([ndx1, ndx2])

        assert merged.n_models == 2
        assert merged.n_tests == 3
        assert set(merged.model_names) == {'m1', 'm2'}
        assert set(merged.test_names) == {'t1', 't2', 't3'}

    def test_merge_preserves_trials(self):
        """Test that merge preserves trial information."""
        mask1 = np.array([[True, False]])
        mask2 = np.array([[False, True]])
        ndx1 = Ndx(['m1'], ['t1', 't2'], mask1)
        ndx2 = Ndx(['m2'], ['t1', 't2'], mask2)

        merged = Ndx.merge([ndx1, ndx2])

        # Check that trials from both are present
        assert merged.trial_mask[0, 0]  # m1, t1 from ndx1
        assert merged.trial_mask[1, 1]  # m2, t2 from ndx2
        assert not merged.trial_mask[0, 1]  # m1, t2 not in ndx1
        assert not merged.trial_mask[1, 0]  # m2, t1 not in ndx2
