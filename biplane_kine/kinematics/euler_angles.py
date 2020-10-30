import numpy as np


def zxy_intrinsic(mat):
    """Return zxy intrinsic Euler angle decomposition of mat (.., 4, 4))"""
    r00 = mat[..., 0, 0]
    r01 = mat[..., 0, 1]
    r02 = mat[..., 0, 2]
    r11 = mat[..., 1, 1]
    r20 = mat[..., 2, 0]
    r21 = mat[..., 2, 1]
    r22 = mat[..., 2, 2]
    theta_z = np.where(r21 < 1, np.where(r21 > -1, np.arctan2(-r01, r11), -np.arctan2(r02, r00)), np.arctan2(r02, r00))
    theta_x = np.where(r21 < 1, np.where(r21 > -1, np.arcsin(r21), -np.pi/2), np.pi/2)
    theta_y = np.where(r21 < 1, np.where(r21 > -1, np.arctan2(-r20, r22), 0), 0)
    return np.stack((theta_z, theta_x, theta_y), -1)
