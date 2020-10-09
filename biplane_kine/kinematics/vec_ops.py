"""This module contains utilities for performing common vector operations."""
import numpy as np


def extended_dot(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """Peform the dot product of the N vectors in vecs1 (N,x) with the corresponding vectors in vecs2 (N, x)."""

    # because the array dimensions must be changed to use matrix multiplication to perform this operation, einsum is
    # (as far as I know) the fastest way to perform this operation
    return np.einsum('ij,ij->i', vecs1, vecs2)
