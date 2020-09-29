import numpy as np


def extended_dot(vecs1, vecs2):
    return np.einsum('ij,ij->i', vecs1, vecs2)
