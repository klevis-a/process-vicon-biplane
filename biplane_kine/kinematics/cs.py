"""This module provides utilities for performing common coordinate system operations."""
import numpy as np


def ht_r(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Combine r (rotation) and t (translation) into a (..., 4, 4) homogeneous matrix (numpy ndarray)."""
    new_shape = list(r.shape)
    new_shape[-1] = new_shape[-1] + 1
    new_shape[-2] = new_shape[-2] + 1
    mat = np.zeros(new_shape)
    mat[..., 3, 3] = 1
    mat[..., :3, :3] = r
    mat[..., :3, 3] = t
    return mat


def ht_inv(mat: np.ndarray) -> np.ndarray:
    """Return the inverse of the homogeneous transformation in mat."""
    mat_inv = np.zeros_like(mat)
    mat_inv[..., 3, 3] = 1
    mat_inv[..., :3, :3] = np.swapaxes(mat[..., :3, :3], -1, -2)
    mat_inv[..., :3, 3] = np.squeeze(-mat_inv[..., :3, :3] @ mat[..., :3, 3, np.newaxis])
    return mat_inv


def vec_transform(t: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Transform the N vectors in vecs (N, 4) using the homogeneous matrix (4, 4) t."""
    # (t @ vecs.T).T = vecs @ t.T
    return vecs @ t.T


def make_vec_hom(vec: np.ndarray) -> np.ndarray:
    """Transform the N vectors in vec (N, 3) into homogeneous counterparts (N, 4)."""
    return np.concatenate((vec, np.ones((vec.shape[0], 1))), axis=1)


def change_cs(new_frame: np.ndarray, frame_traj: np.ndarray) -> np.ndarray:
    """Change all frames in frame_traj (N, 4, 4) into the reference CS specified by new_frame (either (4, 4) or
    (N, 4, 4)."""
    return new_frame @ frame_traj


def pos_ht(mats: np.ndarray) -> np.ndarray:
    """Extract position from homogeneous transformations in mats (N, 4, 4)."""
    return mats[..., :3, 3]


def r_ht(mats: np.ndarray) -> np.ndarray:
    """Extract orientation from homogeneous transformations in mats (N, 4, 4)."""
    return mats[..., :3, :3]
