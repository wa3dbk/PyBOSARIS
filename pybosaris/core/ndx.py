"""
Ndx (Index) class for PyBOSARIS.

The Ndx class encodes trial index information with lists of model names,
test segment names, and a boolean matrix indicating which combinations
are trials of interest.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path

from ..utils.validation import validate_trial_mask, validate_model_test_names


class Ndx:
    """
    Trial index (Ndx) for binary classification experiments.

    An Ndx specifies which model-testsegment pairs constitute trials. It consists of:
    - A list of model names
    - A list of test segment names
    - A boolean matrix indicating which (model, test_segment) pairs are trials

    Parameters
    ----------
    model_names : list of str, optional
        List of model names
    test_names : list of str, optional
        List of test segment names
    trial_mask : ndarray, optional
        Boolean array of shape (n_models, n_tests) indicating valid trials.
        If not provided, all combinations are considered trials.

    Attributes
    ----------
    model_names : list of str
        List of model names
    test_names : list of str
        List of test segment names
    trial_mask : ndarray
        Boolean array indicating which (model, test) pairs are trials

    Examples
    --------
    >>> # Create empty Ndx
    >>> ndx = Ndx()

    >>> # Create with all combinations as trials
    >>> ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'])

    >>> # Create with specific trial mask
    >>> mask = np.array([[True, False, True], [False, True, True]])
    >>> ndx = Ndx(['m1', 'm2'], ['t1', 't2', 't3'], mask)

    >>> # Access properties
    >>> print(ndx.n_models)  # 2
    >>> print(ndx.n_tests)   # 3
    >>> print(ndx.n_trials)  # 4

    Notes
    -----
    The Ndx class is the base class for Key (which adds target/non-target labels).
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        test_names: Optional[List[str]] = None,
        trial_mask: Optional[np.ndarray] = None
    ):
        """Initialize Ndx object."""
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

        # Set up trial mask
        if trial_mask is None:
            # All combinations are trials by default
            self.trial_mask = np.ones(
                (len(model_names), len(test_names)),
                dtype=bool
            )
        else:
            trial_mask = np.asarray(trial_mask, dtype=bool)
            expected_shape = (len(model_names), len(test_names))
            validate_trial_mask(trial_mask, expected_shape)
            self.trial_mask = trial_mask

    @property
    def n_models(self) -> int:
        """Number of models."""
        return len(self.model_names)

    @property
    def n_tests(self) -> int:
        """Number of test segments."""
        return len(self.test_names)

    @property
    def n_trials(self) -> int:
        """Total number of trials (True entries in trial_mask)."""
        return int(np.sum(self.trial_mask))

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of trial mask (n_models, n_tests)."""
        return self.trial_mask.shape

    def __repr__(self) -> str:
        """String representation of Ndx."""
        return (
            f"Ndx({self.n_models} models, {self.n_tests} tests, "
            f"{self.n_trials} trials)"
        )

    def __eq__(self, other: 'Ndx') -> bool:
        """Check equality with another Ndx."""
        if not isinstance(other, Ndx):
            return False
        return (
            self.model_names == other.model_names
            and self.test_names == other.test_names
            and np.array_equal(self.trial_mask, other.trial_mask)
        )

    def validate(self) -> bool:
        """
        Validate Ndx structure.

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

        # Check trial mask
        expected_shape = (len(self.model_names), len(self.test_names))
        validate_trial_mask(self.trial_mask, expected_shape)

        return True

    def filter(
        self,
        model_list: Optional[List[str]] = None,
        test_list: Optional[List[str]] = None,
        keep: bool = True
    ) -> 'Ndx':
        """
        Filter Ndx to keep or remove specified models/tests.

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
        Ndx
            Filtered Ndx object

        Examples
        --------
        >>> ndx = Ndx(['m1', 'm2', 'm3'], ['t1', 't2', 't3'])
        >>> # Keep only certain models
        >>> ndx_filt = ndx.filter(model_list=['m1', 'm2'], keep=True)
        >>> ndx_filt.model_names
        ['m1', 'm2']

        >>> # Remove certain tests
        >>> ndx_filt = ndx.filter(test_list=['t3'], keep=False)
        >>> ndx_filt.test_names
        ['t1', 't2']
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

        # Filter names and mask
        new_model_names = [self.model_names[i] for i in model_indices]
        new_test_names = [self.test_names[i] for i in test_indices]

        # Filter trial mask using numpy indexing
        new_trial_mask = self.trial_mask[np.ix_(model_indices, test_indices)]

        return Ndx(new_model_names, new_test_names, new_trial_mask)

    @classmethod
    def merge(cls, ndx_list: List['Ndx']) -> 'Ndx':
        """
        Merge multiple Ndx objects.

        Parameters
        ----------
        ndx_list : list of Ndx
            List of Ndx objects to merge

        Returns
        -------
        Ndx
            Merged Ndx object

        Raises
        ------
        ValueError
            If Ndx objects have conflicting information

        Examples
        --------
        >>> ndx1 = Ndx(['m1'], ['t1', 't2'])
        >>> ndx2 = Ndx(['m2'], ['t1', 't2', 't3'])
        >>> ndx_merged = Ndx.merge([ndx1, ndx2])
        >>> ndx_merged.model_names
        ['m1', 'm2']
        >>> ndx_merged.test_names
        ['t1', 't2', 't3']

        Notes
        -----
        - Model and test segment lists are combined (union)
        - If a (model, test) pair appears in multiple Ndx objects as a trial,
          it is marked as a trial in the merged result
        - If a (model, test) pair does not appear in any input, it is marked False
        """
        if not ndx_list:
            return cls()

        if len(ndx_list) == 1:
            return ndx_list[0]

        # Collect all unique model and test names
        all_models = []
        all_tests = []
        for ndx in ndx_list:
            all_models.extend(ndx.model_names)
            all_tests.extend(ndx.test_names)

        # Get unique names while preserving order
        unique_models = []
        for name in all_models:
            if name not in unique_models:
                unique_models.append(name)

        unique_tests = []
        for name in all_tests:
            if name not in unique_tests:
                unique_tests.append(name)

        # Create merged trial mask
        merged_mask = np.zeros((len(unique_models), len(unique_tests)), dtype=bool)

        # Fill in mask from each Ndx
        for ndx in ndx_list:
            for i, model in enumerate(ndx.model_names):
                for j, test in enumerate(ndx.test_names):
                    if ndx.trial_mask[i, j]:
                        mi = unique_models.index(model)
                        tj = unique_tests.index(test)
                        merged_mask[mi, tj] = True

        return cls(unique_models, unique_tests, merged_mask)

    def copy(self) -> 'Ndx':
        """
        Create a deep copy of the Ndx.

        Returns
        -------
        Ndx
            Copy of this Ndx
        """
        return Ndx(
            self.model_names.copy(),
            self.test_names.copy(),
            self.trial_mask.copy()
        )

    def to_dict(self) -> dict:
        """
        Convert Ndx to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            'model_names': self.model_names,
            'test_names': self.test_names,
            'trial_mask': self.trial_mask
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Ndx':
        """
        Create Ndx from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with 'model_names', 'test_names', 'trial_mask' keys

        Returns
        -------
        Ndx
            Ndx object
        """
        return cls(
            model_names=data['model_names'],
            test_names=data['test_names'],
            trial_mask=data['trial_mask']
        )

    # I/O methods
    def save(self, filename: Union[str, Path], compression: Optional[str] = 'gzip') -> None:
        """
        Save Ndx to HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to output file (should have .h5 or .hdf5 extension)
        compression : str, optional
            Compression algorithm ('gzip', 'lzf', or None). Default: 'gzip'

        Examples
        --------
        >>> ndx.save('ndx.h5')

        Notes
        -----
        Requires h5py package. Install with: pip install h5py
        """
        from ..io import save_ndx_hdf5
        save_ndx_hdf5(self, str(filename), compression=compression)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'Ndx':
        """
        Load Ndx from HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to HDF5 file

        Returns
        -------
        Ndx
            Loaded Ndx object

        Examples
        --------
        >>> ndx = Ndx.load('ndx.h5')

        Notes
        -----
        Requires h5py package. Install with: pip install h5py
        """
        from ..io import load_ndx_hdf5
        return load_ndx_hdf5(str(filename))
