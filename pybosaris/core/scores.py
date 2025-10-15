"""
Scores class for PyBOSARIS.

The Scores class represents score matrices for binary classification trials.
It stores scores along with model/test segment names and a mask indicating
which scores are valid.
"""

import numpy as np
from typing import List, Optional, Tuple, Union, Callable
from pathlib import Path
import warnings

from .ndx import Ndx
from .key import Key
from ..utils.validation import (
    validate_trial_mask,
    validate_model_test_names,
    validate_scores
)


class Scores:
    """
    Score storage for binary classification experiments.

    A Scores object stores scores for model-test segment pairs in a matrix
    format, along with a boolean mask indicating which scores are valid.

    Parameters
    ----------
    model_names : list of str, optional
        List of model names
    test_names : list of str, optional
        List of test segment names
    score_mat : ndarray, optional
        Score matrix of shape (n_models, n_tests)
    score_mask : ndarray, optional
        Boolean array of shape (n_models, n_tests) indicating valid scores.
        If not provided, all scores are assumed valid.

    Attributes
    ----------
    model_names : list of str
        List of model names
    test_names : list of str
        List of test segment names
    score_mat : ndarray
        Score matrix
    score_mask : ndarray
        Boolean mask indicating valid scores

    Examples
    --------
    >>> # Create empty Scores
    >>> scores = Scores()

    >>> # Create with score matrix
    >>> score_mat = np.array([[1.5, -0.3], [0.8, 2.1]])
    >>> scores = Scores(['m1', 'm2'], ['t1', 't2'], score_mat)

    >>> # Create with custom mask
    >>> mask = np.array([[True, False], [True, True]])
    >>> scores = Scores(['m1', 'm2'], ['t1', 't2'], score_mat, mask)

    >>> # Access properties
    >>> print(scores.n_models)  # 2
    >>> print(scores.n_tests)   # 2
    >>> print(scores.n_scores)  # 3 (only where mask is True)

    Notes
    -----
    - Scores where score_mask is False are ignored in most operations
    - The score_mat can contain any finite values where score_mask is True
    - Values where score_mask is False are typically set to 0 but are ignored
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        test_names: Optional[List[str]] = None,
        score_mat: Optional[np.ndarray] = None,
        score_mask: Optional[np.ndarray] = None
    ):
        """Initialize Scores object."""
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

        # Set up score matrix
        shape = (len(model_names), len(test_names))

        if score_mat is None:
            self.score_mat = np.zeros(shape, dtype=float)
        else:
            score_mat = np.asarray(score_mat, dtype=float)
            if score_mat.shape != shape:
                raise ValueError(
                    f"score_mat shape {score_mat.shape} does not match "
                    f"expected shape {shape} from model/test names"
                )
            self.score_mat = score_mat

        # Set up score mask
        if score_mask is None:
            # All scores valid by default
            self.score_mask = np.ones(shape, dtype=bool)
        else:
            score_mask = np.asarray(score_mask, dtype=bool)
            validate_trial_mask(score_mask, shape)
            self.score_mask = score_mask

        # Validate scores where mask is True
        if self.score_mask.any():
            validate_scores(self.score_mat[self.score_mask])

    @property
    def n_models(self) -> int:
        """Number of models."""
        return len(self.model_names)

    @property
    def n_tests(self) -> int:
        """Number of test segments."""
        return len(self.test_names)

    @property
    def n_scores(self) -> int:
        """Number of valid scores (True entries in score_mask)."""
        return int(np.sum(self.score_mask))

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of score matrix (n_models, n_tests)."""
        return self.score_mat.shape

    def __repr__(self) -> str:
        """String representation of Scores."""
        return (
            f"Scores({self.n_models} models, {self.n_tests} tests, "
            f"{self.n_scores} scores)"
        )

    def __eq__(self, other: 'Scores') -> bool:
        """Check equality with another Scores."""
        if not isinstance(other, Scores):
            return False
        return (
            self.model_names == other.model_names
            and self.test_names == other.test_names
            and np.array_equal(self.score_mat, other.score_mat)
            and np.array_equal(self.score_mask, other.score_mask)
        )

    def validate(self) -> bool:
        """
        Validate Scores structure.

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

        # Check matrix shapes
        shape = (len(self.model_names), len(self.test_names))
        if self.score_mat.shape != shape:
            raise ValueError(
                f"score_mat shape {self.score_mat.shape} does not match "
                f"expected shape {shape}"
            )
        validate_trial_mask(self.score_mask, shape)

        # Check scores are valid where mask is True
        if self.score_mask.any():
            validate_scores(self.score_mat[self.score_mask])

        return True

    def get_tar_non(self, key: Key) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get target and non-target scores based on a Key.

        Parameters
        ----------
        key : Key
            Key object with target/non-target labels

        Returns
        -------
        tar_scores : ndarray
            1D array of target scores
        non_scores : ndarray
            1D array of non-target scores

        Examples
        --------
        >>> tar = np.array([[True, False], [False, False]])
        >>> non = np.array([[False, True], [True, True]])
        >>> key = Key(['m1', 'm2'], ['t1', 't2'], tar, non)
        >>> scores = Scores(['m1', 'm2'], ['t1', 't2'],
        ...                 np.array([[1.5, -0.3], [0.8, 2.1]]))
        >>> tar_scores, non_scores = scores.get_tar_non(key)
        >>> len(tar_scores), len(non_scores)
        (1, 3)
        """
        # Align scores with key
        aligned = self.align_with_ndx(key)

        # Extract target scores where key.tar_mask is True AND score_mask is True
        tar_mask = key.tar_mask & aligned.score_mask
        tar_scores = aligned.score_mat[tar_mask]

        # Extract non-target scores
        non_mask = key.non_mask & aligned.score_mask
        non_scores = aligned.score_mat[non_mask]

        return tar_scores, non_scores

    def align_with_ndx(self, ndx: Union[Ndx, Key]) -> 'Scores':
        """
        Align Scores with an Ndx or Key object.

        Reorders and resizes the Scores object to match the model/test
        segment ordering in the ndx. This is useful for ensuring multiple
        Scores objects are directly comparable.

        Parameters
        ----------
        ndx : Ndx or Key
            Index object to align with

        Returns
        -------
        Scores
            Aligned Scores object with same ordering as ndx

        Examples
        --------
        >>> # Create scores with different ordering
        >>> scores = Scores(['m2', 'm1'], ['t2', 't1'],
        ...                 np.array([[1, 2], [3, 4]]))
        >>> ndx = Ndx(['m1', 'm2'], ['t1', 't2'])
        >>> aligned = scores.align_with_ndx(ndx)
        >>> aligned.model_names
        ['m1', 'm2']
        >>> aligned.test_names
        ['t1', 't2']

        Notes
        -----
        - Models/tests in ndx but not in self will have scores set to 0
          and score_mask set to False
        - Models/tests in self but not in ndx will be dropped
        - Warnings are issued if models/tests are missing
        """
        # Validate inputs
        if not isinstance(ndx, (Ndx, Key)):
            raise TypeError("ndx must be an Ndx or Key object")

        # Create aligned scores with ndx's model/test names
        aligned = Scores()
        aligned.model_names = list(ndx.model_names)
        aligned.test_names = list(ndx.test_names)

        n_models = len(ndx.model_names)
        n_tests = len(ndx.test_names)

        # Initialize with zeros and False mask
        aligned.score_mat = np.zeros((n_models, n_tests), dtype=float)
        aligned.score_mask = np.zeros((n_models, n_tests), dtype=bool)

        # Find which models and tests are present in both
        # Build mapping from ndx names to self names
        model_map = {}
        for i, name in enumerate(self.model_names):
            model_map[name] = i

        test_map = {}
        for i, name in enumerate(self.test_names):
            test_map[name] = i

        # Fill in scores where both model and test are present
        for i, model in enumerate(ndx.model_names):
            if model in model_map:
                for j, test in enumerate(ndx.test_names):
                    if test in test_map:
                        src_i = model_map[model]
                        src_j = test_map[test]
                        aligned.score_mat[i, j] = self.score_mat[src_i, src_j]
                        aligned.score_mask[i, j] = self.score_mask[src_i, src_j]

        # Intersect score_mask with ndx trial mask
        if isinstance(ndx, Ndx):
            aligned.score_mask = aligned.score_mask & ndx.trial_mask
        else:  # Key
            # Only keep scores for trials that are labeled
            trial_mask = ndx.tar_mask | ndx.non_mask
            aligned.score_mask = aligned.score_mask & trial_mask

        # Check for missing models/tests and warn
        models_present = [m for m in ndx.model_names if m in model_map]
        if len(models_present) < n_models:
            warnings.warn(
                f"Models reduced from {n_models} to {len(models_present)} "
                f"during alignment"
            )

        tests_present = [t for t in ndx.test_names if t in test_map]
        if len(tests_present) < n_tests:
            warnings.warn(
                f"Test segments reduced from {n_tests} to {len(tests_present)} "
                f"during alignment"
            )

        # For Key objects, check for missing targets/non-targets
        if isinstance(ndx, Key):
            tar_mask = ndx.tar_mask & aligned.score_mask
            n_tar_expected = int(np.sum(ndx.tar_mask))
            n_tar_found = int(np.sum(tar_mask))
            if n_tar_found < n_tar_expected:
                warnings.warn(
                    f"{n_tar_expected - n_tar_found} of {n_tar_expected} "
                    f"targets missing"
                )

            non_mask = ndx.non_mask & aligned.score_mask
            n_non_expected = int(np.sum(ndx.non_mask))
            n_non_found = int(np.sum(non_mask))
            if n_non_found < n_non_expected:
                warnings.warn(
                    f"{n_non_expected - n_non_found} of {n_non_expected} "
                    f"non-targets missing"
                )
        else:
            # For Ndx, check for missing trials
            mask = ndx.trial_mask & aligned.score_mask
            n_trials_expected = int(np.sum(ndx.trial_mask))
            n_trials_found = int(np.sum(mask))
            if n_trials_found < n_trials_expected:
                warnings.warn(
                    f"{n_trials_expected - n_trials_found} of {n_trials_expected} "
                    f"trials missing"
                )

        return aligned

    def transform(self, func: Callable[[np.ndarray], np.ndarray]) -> 'Scores':
        """
        Transform scores by applying a function.

        Parameters
        ----------
        func : callable
            Function to apply to scores. Should accept and return numpy arrays.

        Returns
        -------
        Scores
            New Scores object with transformed scores

        Examples
        --------
        >>> scores = Scores(['m1'], ['t1', 't2'], np.array([[1.0, 2.0]]))
        >>> # Apply logit transformation
        >>> from pybosaris.utils import logit
        >>> transformed = scores.transform(lambda x: x * 2)
        >>> transformed.score_mat
        array([[2., 4.]])

        Notes
        -----
        The function is only applied to scores where score_mask is True
        """
        new_scores = self.copy()
        # Apply function only to valid scores
        mask = self.score_mask
        new_scores.score_mat[mask] = func(self.score_mat[mask])

        # Validate transformed scores
        if mask.any():
            validate_scores(new_scores.score_mat[mask])

        return new_scores

    def filter(
        self,
        model_list: Optional[List[str]] = None,
        test_list: Optional[List[str]] = None,
        keep: bool = True
    ) -> 'Scores':
        """
        Filter Scores to keep or remove specified models/tests.

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
        Scores
            Filtered Scores object

        Examples
        --------
        >>> score_mat = np.array([[1, 2, 3], [4, 5, 6]])
        >>> scores = Scores(['m1', 'm2'], ['t1', 't2', 't3'], score_mat)
        >>> scores_filt = scores.filter(model_list=['m1'], keep=True)
        >>> scores_filt.n_models
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

        # Filter names and matrices
        new_model_names = [self.model_names[i] for i in model_indices]
        new_test_names = [self.test_names[i] for i in test_indices]

        # Filter matrices using numpy indexing
        new_score_mat = self.score_mat[np.ix_(model_indices, test_indices)]
        new_score_mask = self.score_mask[np.ix_(model_indices, test_indices)]

        return Scores(new_model_names, new_test_names, new_score_mat, new_score_mask)

    @classmethod
    def merge(cls, scores_list: List['Scores']) -> 'Scores':
        """
        Merge multiple Scores objects.

        Parameters
        ----------
        scores_list : list of Scores
            List of Scores objects to merge

        Returns
        -------
        Scores
            Merged Scores object

        Raises
        ------
        ValueError
            If Scores objects have conflicting scores for the same trial

        Examples
        --------
        >>> scores1 = Scores(['m1'], ['t1', 't2'],
        ...                  np.array([[1.0, 2.0]]))
        >>> scores2 = Scores(['m2'], ['t1', 't2'],
        ...                  np.array([[3.0, 4.0]]))
        >>> scores_merged = Scores.merge([scores1, scores2])
        >>> scores_merged.n_models
        2

        Notes
        -----
        - Model and test segment lists are combined (union)
        - It is an error if two Scores objects have a score for the same trial
        - Where score_mask is False, no conflict is raised
        """
        if not scores_list:
            return cls()

        if len(scores_list) == 1:
            return scores_list[0]

        # Start with empty Scores
        merged = cls()

        for scores in scores_list:
            if merged.n_models == 0 and merged.n_tests == 0:
                # First scores object
                merged = scores.copy()
                continue

            # Get union of model and test names
            all_models = list(merged.model_names)
            for name in scores.model_names:
                if name not in all_models:
                    all_models.append(name)

            all_tests = list(merged.test_names)
            for name in scores.test_names:
                if name not in all_tests:
                    all_tests.append(name)

            # Create expanded matrices
            n_models = len(all_models)
            n_tests = len(all_tests)
            new_score_mat = np.zeros((n_models, n_tests), dtype=float)
            new_score_mask = np.zeros((n_models, n_tests), dtype=bool)

            # Fill in from merged
            for i, model in enumerate(merged.model_names):
                mi = all_models.index(model)
                for j, test in enumerate(merged.test_names):
                    tj = all_tests.index(test)
                    new_score_mat[mi, tj] = merged.score_mat[i, j]
                    new_score_mask[mi, tj] = merged.score_mask[i, j]

            # Fill in from current scores object
            for i, model in enumerate(scores.model_names):
                mi = all_models.index(model)
                for j, test in enumerate(scores.test_names):
                    tj = all_tests.index(test)

                    # Check for conflicts
                    if new_score_mask[mi, tj] and scores.score_mask[i, j]:
                        raise ValueError(
                            f"Conflicting scores for ({model}, {test}): "
                            f"cannot merge Scores objects with overlapping trials"
                        )

                    # Add score (either new or adds to zero)
                    new_score_mat[mi, tj] += scores.score_mat[i, j]
                    new_score_mask[mi, tj] = new_score_mask[mi, tj] | scores.score_mask[i, j]

            # Update merged
            merged = cls(all_models, all_tests, new_score_mat, new_score_mask)

        return merged

    def copy(self) -> 'Scores':
        """
        Create a deep copy of the Scores.

        Returns
        -------
        Scores
            Copy of this Scores
        """
        return Scores(
            self.model_names.copy(),
            self.test_names.copy(),
            self.score_mat.copy(),
            self.score_mask.copy()
        )

    def to_dict(self) -> dict:
        """
        Convert Scores to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary representation
        """
        return {
            'model_names': self.model_names,
            'test_names': self.test_names,
            'score_mat': self.score_mat,
            'score_mask': self.score_mask
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Scores':
        """
        Create Scores from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with required keys

        Returns
        -------
        Scores
            Scores object
        """
        return cls(
            model_names=data['model_names'],
            test_names=data['test_names'],
            score_mat=data['score_mat'],
            score_mask=data['score_mask']
        )

    # I/O methods
    def save(self, filename: Union[str, Path], compression: Optional[str] = 'gzip') -> None:
        """
        Save Scores to HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to output file (should have .h5 or .hdf5 extension)
        compression : str, optional
            Compression algorithm ('gzip', 'lzf', or None). Default: 'gzip'

        Examples
        --------
        >>> scores.save('scores.h5')

        Notes
        -----
        Requires h5py package. Install with: pip install h5py
        """
        from ..io import save_scores_hdf5
        save_scores_hdf5(self, str(filename), compression=compression)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'Scores':
        """
        Load Scores from HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to HDF5 file

        Returns
        -------
        Scores
            Loaded Scores object

        Examples
        --------
        >>> scores = Scores.load('scores.h5')

        Notes
        -----
        Requires h5py package. Install with: pip install h5py
        """
        from ..io import load_scores_hdf5
        return load_scores_hdf5(str(filename))
