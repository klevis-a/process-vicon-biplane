"""This module provides utilities for performing common coordinate system operations."""
import numpy as np


def ht_r(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Combine r (rotation) and t (translation) into a (4, 4) homogeneous matrix (numpy ndarray)."""
    mat = np.zeros((4, 4))
    mat[3, 3] = 1
    mat[:3, :3] = r
    mat[:3, 3] = t
    return mat


def vec_transform(t: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Transform the N vectors in vecs (N, 4) using the homogeneous matrix (4, 4) t."""
    # (t @ vecs.T).T = vecs @ t.T
    return vecs @ t.T


def make_vec_hom(vec: np.ndarray) -> np.ndarray:
    """Transform the N vectors in vec (N, 3) into homogeneous counterparts (N, 4)."""
    return np.concatenate((vec, np.ones((vec.shape[0], 1))), axis=1)
