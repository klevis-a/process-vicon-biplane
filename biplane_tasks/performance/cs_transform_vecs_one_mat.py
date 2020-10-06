import numpy as np
import time


# transform all vectors in vecs into frame mat
# assume that mat is 4x4 and vecs is nx4, and the output should be nx4

# implemented using matrix multiplication
def cs_transform_vecs_one_mat_matmul(mat, vecs):
    # (mat @ vecs.T).T = vecs @ mat.T
    return vecs @ mat.T


# implemented using einsum
def cs_transform_vecs_one_mat_einsum(mat, vecs):
    return np.einsum('ij,kj->ki', mat, vecs)


num_el = 1000
m = np.eye(4) * 2.1
vs = np.arange(num_el).reshape((num_el, 1)) + np.arange(4) + 0.1

matmul_res = cs_transform_vecs_one_mat_matmul(m, vs)
einsum_res = cs_transform_vecs_one_mat_einsum(m, vs)
assert(np.array_equal(matmul_res, einsum_res))

n = 1000

m_r = [np.random.rand(4, 4) for i in range(n)]
vs_r = [np.random.rand(num_el, 4) for i in range(n)]

t0 = time.time()
for i in range(n):
    cs_transform_vecs_one_mat_matmul(m_r[i], vs_r[i])
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

t0 = time.time()
for i in range(n):
    cs_transform_vecs_one_mat_einsum(m_r[i], vs_r[i])
t1 = time.time()
einsum_total_time = (t1-t0)*1000
einsum_avg_time = einsum_total_time / n

print('Matrix multiplication total time {:0.2f} and average time {:0.5f} ms'.format(matmul_total_time, matmul_avg_time))
print('Einsum total time {:0.2f} and average time {:0.5f} ms'.format(einsum_total_time, einsum_avg_time))
