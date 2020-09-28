import numpy as np
import time


# transform all vecs using frames in mats1
# assume that mats is nx4x4 and vecs is nx4x1, and the output should be nx4x1

# implemented using homogenous matrix multiplication
def cs_transform_mats_vecs_ht(mats, vecs):
    return mats @ vecs


# rots are nx3x3, trans are nx3x1, and vecs are nx3x1
# implemented using rotations and translations separately
def cs_transform_mats_vecs_rot_trans(rots, trans, vecs):
    return rots @ vecs + trans


num_el = 1000
ms = np.stack([np.eye(4) * i for i in range(1, 11)], axis=0)
ms[..., 3] = 1
vs = np.arange(10).reshape((10, 1)) + np.arange(4) + 0.1
vs[:, 3] = 1
vs = vs[..., np.newaxis]
r = ms[:, :3, :3]
t = ms[:, :3, 3]
t = t[..., np.newaxis]

matmul_res = cs_transform_mats_vecs_ht(ms, vs)
rt_res = cs_transform_mats_vecs_rot_trans(r, t, vs[:, :3, :])
assert(np.allclose(matmul_res[:, :3, :], rt_res))

n = 1000

ms_r = [np.random.rand(num_el, 4, 4) for i in range(n)]
vs1_r = [np.random.rand(num_el, 4)[..., np.newaxis] for i in range(n)]
vs2_r = [np.random.rand(num_el, 4)[..., np.newaxis] for i in range(n)]
vs1_r_trans = [v[:, :3, :] for v in vs1_r]
vs2_r_trans = [v[:, :3, :] for v in vs2_r]
for i in range(n):
    ms_r[i][:, 3, :] = np.asarray([0, 0, 0, 1])
    vs1_r[i][:, 3, :] = 1
    vs2_r[i][:, 3, :] = 1
r_r = [mat[:, :3, :3] for mat in ms_r]
t_r = [mat[:, :3, 3][..., np.newaxis] for mat in ms_r]


t0 = time.time()
for i in range(n):
    res1 = cs_transform_mats_vecs_ht(ms_r[i], vs1_r[i])
    res2 = cs_transform_mats_vecs_ht(ms_r[i], vs2_r[i])
    res = res1[:, :3] + res2[:, :3]
t1 = time.time()
matmul_total_time = (t1-t0)*1000
matmul_avg_time = matmul_total_time / n

t0 = time.time()
for i in range(n):
    res1 = cs_transform_mats_vecs_rot_trans(r_r[i], t_r[i], vs1_r_trans[i])
    res2 = cs_transform_mats_vecs_rot_trans(r_r[i], t_r[i], vs2_r_trans[i])
    res = res1 + res2
t1 = time.time()
trans_total_time = (t1-t0)*1000
trans_avg_time = trans_total_time / n

print('Homogeneous transformation total time {:0.2f} and average time {:0.5f} ms'.format(matmul_total_time,
                                                                                         matmul_avg_time))
print('Rotation/translation total time {:0.2f} and average time {:0.5f} ms'.format(trans_total_time, trans_avg_time))
