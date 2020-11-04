import numpy as np


def zxy_intrinsic(mat: np.ndarray) -> np.ndarray:
    """Return zxy intrinsic Euler angle decomposition of mat (.., 4, 4))"""
    # let's not consider cases where we have NaNs
    not_nan = ~np.any(np.isnan(mat), (-2, -1))

    # extract needed matrix components
    r00 = mat[not_nan, 0, 0]
    r01 = mat[not_nan, 0, 1]
    r02 = mat[not_nan, 0, 2]
    r11 = mat[not_nan, 1, 1]
    r20 = mat[not_nan, 2, 0]
    r21 = mat[not_nan, 2, 1]
    r22 = mat[not_nan, 2, 2]

    # pre-initialize results
    theta_z = np.full(not_nan.shape, np.nan)
    theta_x = np.full(not_nan.shape, np.nan)
    theta_y = np.full(not_nan.shape, np.nan)

    # compute Euler angles
    theta_z[not_nan] = np.where(r21 < 1, np.where(r21 > -1, np.arctan2(-r01, r11), -np.arctan2(r02, r00)),
                                np.arctan2(r02, r00))
    theta_x[not_nan] = np.where(r21 < 1, np.where(r21 > -1, np.arcsin(r21), -np.pi/2), np.pi/2)
    theta_y[not_nan] = np.where(r21 < 1, np.where(r21 > -1, np.arctan2(-r20, r22), 0), 0)

    return np.stack((theta_z, theta_x, theta_y), -1)
