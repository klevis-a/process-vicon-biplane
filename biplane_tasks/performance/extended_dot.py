import numpy as np
import time


# dot product corresponding vectors in matrix vecs1 with vectors in vecs2
# assume that vecs1 is nx4 and vecs2 is nx4, and the output should be nx4

# implemented using matrix multiplication
def extended_dot_matmul(vecs1, vecs2):
    return np.squeeze(vecs1[:, np.newaxis, :] @ vecs2[:, :, np.newaxis])


# implemented using einsum
def extended_dot_einsum(vecs1, vecs2):
    return np.einsum('ij,ij->i', vecs1, vecs2)


# implemented using sum multiply
def extended_dot_sum_multiply(vecs1, vecs2):
    return np.sum(np.multiply(vecs1, vecs2), axis=1)


num_el = 1000
vs1 = np.arange(10).reshape((10, 1)) + np.arange(3)
vs2 = np.arange(10).reshape((10, 1)) + np.arange(3)

matmul_res = extended_dot_matmul(vs1, vs2)
einsum_res = extended_dot_einsum(vs1, vs2)
sum_multiply_res = extended_dot_sum_multiply(vs1, vs2)
assert(np.array_equal(matmul_res, einsum_res))
assert(np.array_equal(matmul_res, sum_multiply_res))

n = 1000

vs1_r = [np.random.rand(num_el, 4) for i in range(n)]
vs2_r = [np.random.rand(num_el, 4) for i in range(n)]

t0 = time.time()
for i in range(n):
    extended_dot_matmul(vs1_r[i], vs2_r[i])
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

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
print('Einsum total time {:0.2f} and average time {:0.5f} ms'.format(einsum_total_time, einsum_avg_time))
print('Sum multiply total time {:0.2f} and average time {:0.5f} ms'.format(sum_multiply_total_time,
                                                                           sum_multiply_avg_time))
