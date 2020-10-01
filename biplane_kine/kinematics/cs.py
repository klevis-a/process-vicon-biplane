import numpy as np


def ht_r(r, t):
    mat = np.zeros((4, 4))
    mat[3, 3] = 1
    mat[:3, :3] = r
    mat[:3, 3] = t
    return mat


def vec_transform(t, vecs):
    # (t @ vecs.T).T = vecs @ t.T
    return vecs @ t.T


def make_vec_hom(vec):
    return np.concatenate((vec, np.ones((vec.shape[0], 1))), axis=1)
