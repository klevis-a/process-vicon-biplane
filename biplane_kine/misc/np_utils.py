"""This module contains various useful numpy utilities."""

import numpy as np
from typing import Tuple


# written by alimanfoo and retrieved from:
# https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
def find_runs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def rms(vec: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the RMS of vec."""
    return np.sqrt(np.mean(np.square(vec), axis=axis))


def nanrms(vec: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the RMS of vec, ignoring NaN values."""
    return np.sqrt(np.nanmean(np.square(vec), axis=axis))


def mae(vec: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the MAE (mean absolute error) of vec."""
    return np.mean(np.absolute(vec), axis=axis)


def nanmae(vec: np.ndarray, axis=0) -> np.ndarray:
    """Compute the MAE (mean absolute error) of vec, ignoring NaN values."""
    return np.nanmean(np.absolute(vec), axis=axis)
