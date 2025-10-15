"""
Input/Output functions for PyBOSARIS.

This module handles reading and writing of score files, keys, and indexes
in various formats.

Formats Supported:
    HDF5: Efficient binary format (primary) - requires h5py

Functions:
    HDF5 I/O (requires h5py):
        save_scores_hdf5: Save Scores to HDF5
        load_scores_hdf5: Load Scores from HDF5
        save_key_hdf5: Save Key to HDF5
        load_key_hdf5: Load Key from HDF5
        save_ndx_hdf5: Save Ndx to HDF5
        load_ndx_hdf5: Load Ndx from HDF5
        load_hdf5: Auto-detect and load from HDF5
"""

from .hdf5_io import (
    save_scores_hdf5,
    load_scores_hdf5,
    save_key_hdf5,
    load_key_hdf5,
    save_ndx_hdf5,
    load_ndx_hdf5,
    load_hdf5,
    HAS_H5PY
)

__all__ = [
    "save_scores_hdf5",
    "load_scores_hdf5",
    "save_key_hdf5",
    "load_key_hdf5",
    "save_ndx_hdf5",
    "load_ndx_hdf5",
    "load_hdf5",
    "HAS_H5PY",
]
