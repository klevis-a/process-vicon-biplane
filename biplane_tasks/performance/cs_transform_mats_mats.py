"""This script tests whether matmul or einsum is faster at multipliying a collection of matrices with a another
 collection (same length) of matrices of the same dimension. In biomechanics this operation is used when expressing
 a segment in the coordinate system of its proximal segment.
"""

import numpy as np
import time


# transform all frames in mats2 using frames in mats1
# assume that mats1 is nx4x4 and mats2 is nx4x4, and the output should be nx4x4

# implemented using matrix multiplication
def cs_transform_mats_mats_matmul(mats1, mats2):
    return mats1 @ mats2


# implemented using einsum
def cs_transform_mats_mats_einsum(mats1, mats2):
    return np.einsum('ijk,ikl->ijl', mats1, mats2)


# simple test that can be easily verified by eye that the expected transformation is actually taking place
ms1 = np.stack([np.eye(4) * i for i in range(1, 11)], axis=0)
ms2 = np.stack([np.eye(4) * i for i in range(1, 11)], axis=0)

matmul_res = cs_transform_mats_mats_matmul(ms1, ms2)
einsum_res = cs_transform_mats_mats_einsum(ms1, ms2)
assert(np.array_equal(matmul_res, einsum_res))

# performance test
num_el = 1000
n = 1000

ms1_r = [np.random.rand(num_el, 4, 4) for i in range(n)]
ms2_r = [np.random.rand(num_el, 4, 4) for i in range(n)]

t0 = time.time()
for i in range(n):
    cs_transform_mats_mats_matmul(ms1_r[i], ms2_r[i])
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

t0 = time.time()
for i in range(n):
    cs_transform_mats_mats_einsum(ms1_r[i], ms2_r[i])
t1 = time.time()
einsum_total_time = (t1-t0)*1000
einsum_avg_time = einsum_total_time / n

print('Matrix multiplication total time {:0.2f} and average time {:0.5f} ms'.format(matmul_total_time, matmul_avg_time))
print('Einsum total time {:0.2f} and average time {:0.5f} ms'.format(einsum_total_time, einsum_avg_time))
