"""This script tests whether to use the built-in linalg.norm operation or einsum to compute the L2 (Euclidean) norm of
a collection of vectors.
"""

import numpy as np
from numpy import linalg
import time


# norm of vectors in vecs1
# assume that vecs1 is nx4

# implemented using linear algebra module
def la_norm(vecs1):
    return linalg.norm(vecs1, axis=1)


# implemented using einsum
def norm_einsum(vecs1, vecs2):
    return np.sqrt(np.einsum('ij,ij->i', vecs1, vecs2))


# simple test that can be easily verified by eye that the expected transformation is actually taking place
vs1 = np.arange(10).reshape((10, 1)) + np.arange(3)

einsum_res = norm_einsum(vs1, vs1)
la_res = la_norm(vs1)
assert(np.array_equal(einsum_res, la_res))


# performance test
num_el = 1000
n = 1000

vs1_r = [np.random.rand(num_el, 3) for i in range(n)]

t0 = time.time()
for i in range(n):
    la_norm(vs1_r[i])
t1 = time.time()
la_total_time = (t1-t0)*1000
la_avg_time = la_total_time / n

t0 = time.time()
for i in range(n):
    norm_einsum(vs1_r[i], vs1_r[i])
t1 = time.time()
einsum_total_time = (t1-t0)*1000
einsum_avg_time = einsum_total_time / n

print('LA norm total time {:0.2f} and average time {:0.5f} ms'.format(la_total_time, la_avg_time))
print('Einsum total time {:0.2f} and average time {:0.5f} ms'.format(einsum_total_time, einsum_avg_time))
