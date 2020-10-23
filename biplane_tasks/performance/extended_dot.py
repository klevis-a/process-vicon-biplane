"""This script tests whether to use matrix multiplication, sum-multiply, or einsum to compute the dot product of two
collections of vectors. This is typically used in the context of finding the norm of a collection of vectors, but is
also used when computing torque, energy, etc.
"""

import numpy as np
import time


# dot product of vectors in vecs1 with corresponding vectors in vecs2
# assume that vecs1 is nx4 and vecs2 is nx4, and the output should be n

# implemented using matrix multiplication
def extended_dot_matmul(vecs1, vecs2):
    return np.squeeze(vecs1[:, np.newaxis, :] @ vecs2[:, :, np.newaxis])


# implemented using matrix multiplication but it is assumed that vecs1 and vecs 2 are nx4x1
def extended_dot_matmul_ext(vecs1, vecs2):
    return np.squeeze(np.swapaxes(vecs1, 1, 2) @ vecs2)


# implemented using einsum
def extended_dot_einsum(vecs1, vecs2):
    return np.einsum('ij,ij->i', vecs1, vecs2)


# implemented using sum multiply
def extended_dot_sum_multiply(vecs1, vecs2):
    return np.sum(np.multiply(vecs1, vecs2), axis=1)


# simple test that can be easily verified by eye that the expected transformation is actually taking place
vs1 = np.arange(10).reshape((10, 1)) + np.arange(3)
vs2 = np.arange(10).reshape((10, 1)) + np.arange(3)

matmul_res = extended_dot_matmul(vs1, vs2)
matmul_ext_res = extended_dot_matmul_ext(vs1[:, :, np.newaxis], vs2[:, :, np.newaxis])
einsum_res = extended_dot_einsum(vs1, vs2)
sum_multiply_res = extended_dot_sum_multiply(vs1, vs2)
assert(np.array_equal(matmul_res, einsum_res))
assert(np.array_equal(matmul_res, sum_multiply_res))
assert(np.array_equal(matmul_res, matmul_ext_res))


# performance test
num_el = 1000
n = 1000

vs1_r = [np.random.rand(num_el, 4) for i in range(n)]
vs2_r = [np.random.rand(num_el, 4) for i in range(n)]
vs1_ext_r = [mat[:, :, np.newaxis] for mat in vs1_r]
vs2_ext_r = [mat[:, :, np.newaxis] for mat in vs2_r]

t0 = time.time()
for i in range(n):
    extended_dot_matmul(vs1_r[i], vs2_r[i])
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

t0 = time.time()
for i in range(n):
    extended_dot_matmul_ext(vs1_ext_r[i], vs2_ext_r[i])
t1 = time.time()
matmul_ext_total_time = (t1-t0)*1000
matmul_ext_avg_time = matmul_ext_total_time / n

t0 = time.time()
for i in range(n):
    extended_dot_einsum(vs1_r[i], vs2_r[i])
t1 = time.time()
einsum_total_time = (t1-t0)*1000
einsum_avg_time = einsum_total_time / n

t0 = time.time()
for i in range(n):
    extended_dot_sum_multiply(vs1_r[i], vs2_r[i])
t1 = time.time()
sum_multiply_total_time = (t1-t0)*1000
sum_multiply_avg_time = sum_multiply_total_time / n

print('Matrix multiplication total time {:0.2f} and average time {:0.5f} ms'.format(matmul_total_time, matmul_avg_time))
print('Matrix multiplication extended total time {:0.2f} and average time {:0.5f} ms'.format(matmul_ext_total_time,
                                                                                             matmul_ext_avg_time))
print('Einsum total time {:0.2f} and average time {:0.5f} ms'.format(einsum_total_time, einsum_avg_time))
print('Sum multiply total time {:0.2f} and average time {:0.5f} ms'.format(sum_multiply_total_time,
                                                                           sum_multiply_avg_time))
