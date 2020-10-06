import numpy as np
import time


# transform all frames in mats into frame mat
# assume that mat is 4x4 and mats is nx4x4, and the output should be nx4x4

# implemented using matrix multiplication
def cs_transform_mats_one_mat_matmul(mat, mats):
    return mat @ mats


# implemented using einsum
def cs_transform_mats_one_mat_einsum(mat, mats):
    return np.einsum('jk,ikl->ijl', mat, mats)


num_el = 1000
m = np.eye(4) * 2.1
ms = np.stack([np.eye(4) * i for i in range(1, 11)], axis=0)

matmul_res = cs_transform_mats_one_mat_matmul(m, ms)
einsum_res = cs_transform_mats_one_mat_einsum(m, ms)
assert(np.array_equal(matmul_res, einsum_res))

n = 1000

m_r = [np.random.rand(4, 4) for i in range(n)]
ms_r = [np.random.rand(num_el, 4, 4) for i in range(n)]

t0 = time.time()
for i in range(n):
    cs_transform_mats_one_mat_matmul(m_r[i], ms_r[i])
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

t0 = time.time()
for i in range(n):
    cs_transform_mats_one_mat_einsum(m_r[i], ms_r[i])
t1 = time.time()
einsum_total_time = (t1-t0)*1000
einsum_avg_time = einsum_total_time / n

print('Matrix multiplication total time {:0.2f} and average time {:0.5f} ms'.format(matmul_total_time, matmul_avg_time))
print('Einsum total time {:0.2f} and average time {:0.5f} ms'.format(einsum_total_time, einsum_avg_time))
