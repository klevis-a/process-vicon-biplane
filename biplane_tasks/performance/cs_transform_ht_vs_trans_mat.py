"""This script tests whether to use homogeneous transformation matrices (4x4), or a rotation matrix (3x3) and
translation vector (3x1) to transform a collection of matrices from one coordinate system to another. In terms of
memory, storing the rotation matrix and translation vector separately is more efficient, but we are almost never memory-
bound when considering the amount of data present in a typical study.
"""

import numpy as np
import time


# transform all frames in mats2 using frames in mats1
# assume that mats1 is nx4x4 and mats2 is nx4x4, and the output should be nx4x4

# implemented using homogenous matrix multiplication
def cs_transform_mats_mats_ht(mats1, mats2):
    return mats1 @ mats2


# rots are nx3x3 and trans are nx3x1
# implemented using rotations and translations separately
def cs_transform_mats_mats_rot_trans(rots1, trans1, rots2, trans2):
    return rots1 @ rots2, rots1 @ trans2 + trans1


# simple test that can be easily verified by eye that the expected transformation is actually taking place
ms1 = np.stack([np.eye(4) * i for i in range(1, 11)], axis=0)
ms2 = np.stack([np.eye(4) * i for i in range(1, 11)], axis=0)
ms1[..., 3] = 1
ms2[..., 3] = 1
r1 = ms1[:, :3, :3]
t1 = ms1[:, :3, 3]
t1 = t1[..., np.newaxis]
r2 = ms2[:, :3, :3]
t2 = ms2[:, :3, 3]
t2 = t2[..., np.newaxis]

matmul_res = cs_transform_mats_mats_ht(ms1, ms2)
r_res, t_res = cs_transform_mats_mats_rot_trans(r1, t1, r2, t2)
assert(np.array_equal(matmul_res[:, :3, :3], r_res))
assert(np.array_equal(matmul_res[:, :3, 3][..., np.newaxis], t_res))


# performance test
n = 1000
num_el = 1000

ms1_r = [np.random.rand(num_el, 4, 4) for i in range(n)]
ms2_r = [np.random.rand(num_el, 4, 4) for i in range(n)]
for i in range(n):
    ms1_r[i][:, 3, :] = np.asarray([0, 0, 0, 1])
    ms2_r[i][:, 3, :] = np.asarray([0, 0, 0, 1])

r1_r = [mat[:, :3, :3] for mat in ms1_r]
t1_r = [mat[:, :3, 3][..., np.newaxis] for mat in ms1_r]
r2_r = [mat[:, :3, :3] for mat in ms2_r]
t2_r = [mat[:, :3, 3][..., np.newaxis] for mat in ms2_r]

t0 = time.time()
for i in range(n):
    cs_transform_mats_mats_ht(ms1_r[i], ms2_r[i])
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

t0 = time.time()
for i in range(n):
    cs_transform_mats_mats_rot_trans(r1_r[i], t1_r[i], r2_r[i], t2_r[i])
t1 = time.time()
trans_total_time = (t1-t0)*1000
trans_avg_time = trans_total_time / n

print('Homogeneous transformation total time {:0.2f} and average time {:0.5f} ms'.format(matmul_total_time,
                                                                                         matmul_avg_time))
print('Rotation/translation total time {:0.2f} and average time {:0.5f} ms'.format(trans_total_time, trans_avg_time))
