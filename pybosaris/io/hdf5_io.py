"""
HDF5 I/O functions for PyBOSARIS.

This module provides functions to save and load PyBOSARIS objects
(Scores, Key, Ndx) to/from HDF5 format.

Note: Requires h5py package. Install with: pip install h5py
"""

import numpy as np
from typing import Union, Optional
import warnings

# Try to import h5py, but make it optional
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None


def _check_h5py():
    """Check if h5py is available and raise helpful error if not."""
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 I/O operations. "
            "Install it with: pip install h5py"
        )


def save_scores_hdf5(scores, filename: str, compression: Optional[str] = 'gzip'):
    """
    Save Scores object to HDF5 file.

    Parameters
    ----------
    scores : Scores
        Scores object to save
    filename : str
        Path to output HDF5 file
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', or None)
        Default: 'gzip'

    Examples
    --------
    >>> from pybosaris.core import Scores
    >>> scores = Scores(['m1', 'm2'], ['t1', 't2'],
    ...                 np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> save_scores_hdf5(scores, 'scores.h5')
    """
    _check_h5py()

    from ..core import Scores
    if not isinstance(scores, Scores):
        raise TypeError(f"Expected Scores object, got {type(scores)}")

    with h5py.File(filename, 'w') as f:
        # Save model names
        f.create_dataset(
            'model_names',
            data=np.array(scores.model_names, dtype='S'),
            compression=compression
        )

        # Save test names
        f.create_dataset(
            'test_names',
            data=np.array(scores.test_names, dtype='S'),
            compression=compression
        )

        # Save score matrix
        f.create_dataset(
            'score_mat',
            data=scores.score_mat,
            compression=compression
        )

        # Save metadata
        f.attrs['type'] = 'Scores'
        f.attrs['version'] = '1.0'


def load_scores_hdf5(filename: str):
    """
    Load Scores object from HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file

    Returns
    -------
    Scores
        Loaded Scores object

    Examples
    --------
    >>> scores = load_scores_hdf5('scores.h5')
    >>> print(scores.shape)
    """
    _check_h5py()

    from ..core import Scores

    with h5py.File(filename, 'r') as f:
        # Verify file type
        if f.attrs.get('type') != 'Scores':
            warnings.warn(
                f"File does not have 'Scores' type marker. "
                f"Found: {f.attrs.get('type')}"
            )

        # Load data
        model_names = [name.decode('utf-8') for name in f['model_names'][:]]
        test_names = [name.decode('utf-8') for name in f['test_names'][:]]
        score_mat = f['score_mat'][:]

    return Scores(model_names, test_names, score_mat)


def save_key_hdf5(key, filename: str, compression: Optional[str] = 'gzip'):
    """
    Save Key object to HDF5 file.

    Parameters
    ----------
    key : Key
        Key object to save
    filename : str
        Path to output HDF5 file
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', or None)

    Examples
    --------
    >>> from pybosaris.core import Key
    >>> key = Key(['m1'], ['t1'], np.array([[True]]), np.array([[False]]))
    >>> save_key_hdf5(key, 'key.h5')
    """
    _check_h5py()

    from ..core import Key
    if not isinstance(key, Key):
        raise TypeError(f"Expected Key object, got {type(key)}")

    with h5py.File(filename, 'w') as f:
        # Save model names
        f.create_dataset(
            'model_names',
            data=np.array(key.model_names, dtype='S'),
            compression=compression
        )

        # Save test names
        f.create_dataset(
            'test_names',
            data=np.array(key.test_names, dtype='S'),
            compression=compression
        )

        # Save masks
        f.create_dataset(
            'tar_mask',
            data=key.tar_mask,
            compression=compression
        )
        f.create_dataset(
            'non_mask',
            data=key.non_mask,
            compression=compression
        )

        # Save metadata
        f.attrs['type'] = 'Key'
        f.attrs['version'] = '1.0'


def load_key_hdf5(filename: str):
    """
    Load Key object from HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file

    Returns
    -------
    Key
        Loaded Key object

    Examples
    --------
    >>> key = load_key_hdf5('key.h5')
    >>> print(key.shape)
    """
    _check_h5py()

    from ..core import Key

    with h5py.File(filename, 'r') as f:
        # Verify file type
        if f.attrs.get('type') != 'Key':
            warnings.warn(
                f"File does not have 'Key' type marker. "
                f"Found: {f.attrs.get('type')}"
            )

        # Load data
        model_names = [name.decode('utf-8') for name in f['model_names'][:]]
        test_names = [name.decode('utf-8') for name in f['test_names'][:]]
        tar_mask = f['tar_mask'][:]
        non_mask = f['non_mask'][:]

    return Key(model_names, test_names, tar_mask, non_mask)


def save_ndx_hdf5(ndx, filename: str, compression: Optional[str] = 'gzip'):
    """
    Save Ndx object to HDF5 file.

    Parameters
    ----------
    ndx : Ndx
        Ndx object to save
    filename : str
        Path to output HDF5 file
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', or None)

    Examples
    --------
    >>> from pybosaris.core import Ndx
    >>> ndx = Ndx(['m1'], ['t1'])
    >>> save_ndx_hdf5(ndx, 'ndx.h5')
    """
    _check_h5py()

    from ..core import Ndx
    if not isinstance(ndx, Ndx):
        raise TypeError(f"Expected Ndx object, got {type(ndx)}")

    with h5py.File(filename, 'w') as f:
        # Save model names
        f.create_dataset(
            'model_names',
            data=np.array(ndx.model_names, dtype='S'),
            compression=compression
        )

        # Save test names
        f.create_dataset(
            'test_names',
            data=np.array(ndx.test_names, dtype='S'),
            compression=compression
        )

        # Save trial mask
        f.create_dataset(
            'trial_mask',
            data=ndx.trial_mask,
            compression=compression
        )

        # Save metadata
        f.attrs['type'] = 'Ndx'
        f.attrs['version'] = '1.0'


def load_ndx_hdf5(filename: str):
    """
    Load Ndx object from HDF5 file.

    Parameters
    ----------
    filename : str
        Path to HDF5 file

    Returns
    -------
    Ndx
        Loaded Ndx object

    Examples
    --------
    >>> ndx = load_ndx_hdf5('ndx.h5')
    >>> print(ndx.shape)
    """
    _check_h5py()

    from ..core import Ndx

    with h5py.File(filename, 'r') as f:
        # Verify file type
        if f.attrs.get('type') != 'Ndx':
            warnings.warn(
                f"File does not have 'Ndx' type marker. "
                f"Found: {f.attrs.get('type')}"
            )

        # Load data
        model_names = [name.decode('utf-8') for name in f['model_names'][:]]
        test_names = [name.decode('utf-8') for name in f['test_names'][:]]
        trial_mask = f['trial_mask'][:]

    return Ndx(model_names, test_names, trial_mask)


# Convenience function for auto-detection
def load_hdf5(filename: str):
    """
    Load PyBOSARIS object from HDF5 file (auto-detect type).

    Parameters
    ----------
    filename : str
        Path to HDF5 file

    Returns
    -------
    Union[Scores, Key, Ndx]
        Loaded object (type depends on file content)

    Raises
    ------
    ValueError
        If file type cannot be determined

    Examples
    --------
    >>> obj = load_hdf5('data.h5')  # Auto-detects type
    """
    _check_h5py()

    with h5py.File(filename, 'r') as f:
        obj_type = f.attrs.get('type')

    if obj_type == 'Scores':
        return load_scores_hdf5(filename)
    elif obj_type == 'Key':
        return load_key_hdf5(filename)
    elif obj_type == 'Ndx':
        return load_ndx_hdf5(filename)
    else:
        raise ValueError(
            f"Unknown object type in HDF5 file: {obj_type}. "
            f"Expected 'Scores', 'Key', or 'Ndx'"
        )
