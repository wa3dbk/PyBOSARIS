"""
Key class for PyBOSARIS.

The Key class represents supervised trial lists with target/non-target labels.
It extends Ndx by adding labels to distinguish target and non-target trials.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path

from .ndx import Ndx
from ..utils.validation import (
    validate_trial_mask,
    validate_model_test_names,
    validate_key_consistency
)


class Key:
    """
    Supervised trial key with target/non-target labels.

    A Key specifies which model-test segment pairs are target trials
    (same source) and which are non-target trials (different sources).

    Parameters
    ----------
    model_names : list of str, optional
        List of model names
    test_names : list of str, optional
        List of test segment names
    target_mask : ndarray, optional
        Boolean array of shape (n_models, n_tests) indicating target trials
    nontarget_mask : ndarray, optional
        Boolean array of shape (n_models, n_tests) indicating non-target trials

    Attributes
    ----------
    model_names : list of str
        List of model names
    test_names : list of str
        List of test segment names
    tar_mask : ndarray
        Boolean array indicating target trials
    non_mask : ndarray
        Boolean array indicating non-target trials

    Examples
    --------
    >>> # Create empty Key
    >>> key = Key()

    >>> # Create with target and non-target masks
    >>> tar = np.array([[True, False], [False, False]])
    >>> non = np.array([[False, True], [True, True]])
    >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar, non)

    >>> # Access properties
    >>> print(key.n_targets)    # 1
    >>> print(key.n_nontargets) # 3

    Notes
    -----
    - A trial can be either target, non-target, or neither (unlabeled)
    - A trial cannot be both target and non-target (validated on construction)
    - Target mask: True where model and test come from same source
    - Non-target mask: True where model and test come from different sources
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        test_names: Optional[List[str]] = None,
        target_mask: Optional[np.ndarray] = None,
        nontarget_mask: Optional[np.ndarray] = None
    ):
        """Initialize Key object."""
        # Handle empty constructor
        if model_names is None:
            model_names = []
        if test_names is None:
            test_names = []

        # Validate inputs
        validate_model_test_names(model_names, test_names)

        # Store names
        self.model_names = list(model_names)
        self.test_names = list(test_names)

        # Set up masks
        shape = (len(model_names), len(test_names))

        if target_mask is None:
            self.tar_mask = np.zeros(shape, dtype=bool)
        else:
            target_mask = np.asarray(target_mask, dtype=bool)
            validate_trial_mask(target_mask, shape)
            self.tar_mask = target_mask

        if nontarget_mask is None:
            self.non_mask = np.zeros(shape, dtype=bool)
        else:
            nontarget_mask = np.asarray(nontarget_mask, dtype=bool)
            validate_trial_mask(nontarget_mask, shape)
            self.non_mask = nontarget_mask

        # Validate no overlap
        validate_key_consistency(self.tar_mask, self.non_mask)

    @property
    def n_models(self) -> int:
        """Number of models."""
        return len(self.model_names)

    @property
    def n_tests(self) -> int:
        """Number of test segments."""
        return len(self.test_names)

    @property
    def n_targets(self) -> int:
        """Number of target trials."""
        return int(np.sum(self.tar_mask))

    @property
    def n_nontargets(self) -> int:
        """Number of non-target trials."""
        return int(np.sum(self.non_mask))

    @property
    def n_trials(self) -> int:
        """Total number of labeled trials."""
        return self.n_targets + self.n_nontargets

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of masks (n_models, n_tests)."""
        return self.tar_mask.shape

    def __repr__(self) -> str:
        """String representation of Key."""
        return (
            f"Key({self.n_models} models, {self.n_tests} tests, "
            f"{self.n_targets} targets, {self.n_nontargets} non-targets)"
        )

    def __eq__(self, other: 'Key') -> bool:
        """Check equality with another Key."""
        if not isinstance(other, Key):
            return False
        return (
            self.model_names == other.model_names
            and self.test_names == other.test_names
            and np.array_equal(self.tar_mask, other.tar_mask)
            and np.array_equal(self.non_mask, other.non_mask)
        )

    def validate(self) -> bool:
        """
        Validate Key structure.

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check names are valid
        validate_model_test_names(self.model_names, self.test_names)

        # Check masks
        shape = (len(self.model_names), len(self.test_names))
        validate_trial_mask(self.tar_mask, shape)
        validate_trial_mask(self.non_mask, shape)

        # Check no overlap
        validate_key_consistency(self.tar_mask, self.non_mask)

        return True

    def to_ndx(self) -> Ndx:
        """
        Convert Key to Ndx (trial mask = target OR non-target).

        Returns
        -------
        Ndx
            Ndx with trials marked wherever key has target or non-target label

        Examples
        --------
        >>> tar = np.array([[True, False], [False, False]])
        >>> non = np.array([[False, True], [True, True]])
        >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar, non)
        >>> ndx = key.to_ndx()
        >>> ndx.n_trials  # 4 (all labeled trials)
        4
        """
        trial_mask = np.logical_or(self.tar_mask, self.non_mask)
        return Ndx(self.model_names, self.test_names, trial_mask)

    def get_tar_non_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get flat arrays of target and non-target scores.

        Returns
        -------
        tar_indices : ndarray
            1D array of indices for target trials (in row-major order)
        non_indices : ndarray
            1D array of indices for non-target trials (in row-major order)

        Examples
        --------
        >>> tar = np.array([[True, False], [False, False]])
        >>> non = np.array([[False, True], [True, True]])
        >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar, non)
        >>> tar_idx, non_idx = key.get_tar_non_arrays()
        >>> len(tar_idx), len(non_idx)
        (1, 3)
        """
        tar_indices = np.where(self.tar_mask.ravel())[0]
        non_indices = np.where(self.non_mask.ravel())[0]
        return tar_indices, non_indices

    def filter(
        self,
        model_list: Optional[List[str]] = None,
        test_list: Optional[List[str]] = None,
        keep: bool = True
    ) -> 'Key':
        """
        Filter Key to keep or remove specified models/tests.

        Parameters
        ----------
        model_list : list of str, optional
            List of model names to filter
        test_list : list of str, optional
            List of test segment names to filter
        keep : bool, optional
            If True, keep only specified models/tests.
            If False, remove specified models/tests (default: True)

        Returns
        -------
        Key
            Filtered Key object

        Examples
        --------
        >>> tar = np.array([[True, False, False], [False, True, False]])
        >>> non = np.array([[False, True, True], [True, False, True]])
        >>> key = Key(['m1', 'm2'], ['t1', 't2', 't3'], tar, non)
        >>> key_filt = key.filter(model_list=['m1'], keep=True)
        >>> key_filt.n_models
        1
        """
        # Determine which models and tests to keep
        if model_list is not None:
            if keep:
                model_indices = [
                    i for i, name in enumerate(self.model_names)
                    if name in model_list
                ]
            else:
                model_indices = [
                    i for i, name in enumerate(self.model_names)
                    if name not in model_list
                ]
        else:
            model_indices = list(range(self.n_models))

        if test_list is not None:
            if keep:
                test_indices = [
                    i for i, name in enumerate(self.test_names)
                    if name in test_list
                ]
            else:
                test_indices = [
                    i for i, name in enumerate(self.test_names)
                    if name not in test_list
                ]
        else:
            test_indices = list(range(self.n_tests))

        # Filter names and masks
        new_model_names = [self.model_names[i] for i in model_indices]
        new_test_names = [self.test_names[i] for i in test_indices]

        # Filter masks using numpy indexing
        new_tar_mask = self.tar_mask[np.ix_(model_indices, test_indices)]
        new_non_mask = self.non_mask[np.ix_(model_indices, test_indices)]

        return Key(new_model_names, new_test_names, new_tar_mask, new_non_mask)

    @classmethod
    def merge(cls, key_list: List['Key']) -> 'Key':
        """
        Merge multiple Key objects.

        Parameters
        ----------
        key_list : list of Key
            List of Key objects to merge

        Returns
        -------
        Key
            Merged Key object

        Raises
        ------
        ValueError
            If Key objects have conflicting labels for the same trial

        Examples
        --------
        >>> tar1 = np.array([[True, False]])
        >>> non1 = np.array([[False, True]])
        >>> key1 = Key(['m1'], ['t1', 't2'], tar1, non1)

        >>> tar2 = np.array([[False, True]])
        >>> non2 = np.array([[True, False]])
        >>> key2 = Key(['m2'], ['t1', 't2'], tar2, non2)

        >>> key_merged = Key.merge([key1, key2])
        >>> key_merged.n_targets
        2
        >>> key_merged.n_nontargets
        2
        """
        if not key_list:
            return cls()

        if len(key_list) == 1:
            return key_list[0]

        # Collect all unique model and test names
        all_models = []
        all_tests = []
        for key in key_list:
            all_models.extend(key.model_names)
            all_tests.extend(key.test_names)

        # Get unique names while preserving order
        unique_models = []
        for name in all_models:
            if name not in unique_models:
                unique_models.append(name)

        unique_tests = []
        for name in all_tests:
            if name not in unique_tests:
                unique_tests.append(name)

        # Create merged masks
        merged_tar = np.zeros((len(unique_models), len(unique_tests)), dtype=bool)
        merged_non = np.zeros((len(unique_models), len(unique_tests)), dtype=bool)

        # Fill in masks from each Key
        for key in key_list:
            for i, model in enumerate(key.model_names):
                for j, test in enumerate(key.test_names):
                    mi = unique_models.index(model)
                    tj = unique_tests.index(test)

                    # Check for conflicts
                    if key.tar_mask[i, j]:
                        if merged_non[mi, tj]:
                            raise ValueError(
                                f"Conflicting labels for ({model}, {test}): "
                                f"marked as both target and non-target"
                            )
                        merged_tar[mi, tj] = True

                    if key.non_mask[i, j]:
                        if merged_tar[mi, tj]:
                            raise ValueError(
                                f"Conflicting labels for ({model}, {test}): "
                                f"marked as both target and non-target"
                            )
                        merged_non[mi, tj] = True

        return cls(unique_models, unique_tests, merged_tar, merged_non)

    def copy(self) -> 'Key':
        """
        Create a deep copy of the Key.

        Returns
        -------
        Key
            Copy of this Key
        """
        return Key(
            self.model_names.copy(),
            self.test_names.copy(),
            self.tar_mask.copy(),
            self.non_mask.copy()
        )

    def to_dict(self) -> dict:
        """
        Convert Key to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            'model_names': self.model_names,
            'test_names': self.test_names,
            'tar_mask': self.tar_mask,
            'non_mask': self.non_mask
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Key':
        """
        Create Key from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with required keys

        Returns
        -------
        Key
            Key object
        """
        return cls(
            model_names=data['model_names'],
            test_names=data['test_names'],
            target_mask=data['tar_mask'],
            nontarget_mask=data['non_mask']
        )

    # I/O methods
    def save(self, filename: Union[str, Path], compression: Optional[str] = 'gzip') -> None:
        """
        Save Key to HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to output file (should have .h5 or .hdf5 extension)
        compression : str, optional
            Compression algorithm ('gzip', 'lzf', or None). Default: 'gzip'

        Examples
        --------
        >>> key.save('key.h5')

        Notes
        -----
        Requires h5py package. Install with: pip install h5py
        """
        from ..io import save_key_hdf5
        save_key_hdf5(self, str(filename), compression=compression)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'Key':
        """
        Load Key from HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to HDF5 file

        Returns
        -------
        Key
            Loaded Key object

        Examples
        --------
        >>> key = Key.load('key.h5')

        Notes
        -----
        Requires h5py package. Install with: pip install h5py
        """
        from ..io import load_key_hdf5
        return load_key_hdf5(str(filename))
