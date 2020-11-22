from typing import Tuple
import numpy as np


def extract_mat_components(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    not_nan = ~np.any(np.isnan(mat), (-2, -1))

    # extract matrix components
    r00 = mat[not_nan, 0, 0]
    r01 = mat[not_nan, 0, 1]
    r02 = mat[not_nan, 0, 2]

    r10 = mat[not_nan, 1, 0]
    r11 = mat[not_nan, 1, 1]
    r12 = mat[not_nan, 1, 2]

    r20 = mat[not_nan, 2, 0]
    r21 = mat[not_nan, 2, 1]
    r22 = mat[not_nan, 2, 2]

    return not_nan, r00, r01, r02, r10, r11, r12, r20, r21, r22


def zxy_intrinsic(mat: np.ndarray) -> np.ndarray:
    """Return zxy intrinsic Euler angle decomposition of mat (.., 4, 4))"""
    # extract components
    not_nan, r00, r01, r02, _, r11, _, r20, r21, r22 = extract_mat_components(mat)

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


def yxz_intrinsic(mat: np.ndarray) -> np.ndarray:
    """Return yxz intrinsic Euler angle decomposition of mat (.., 4, 4))"""
    # extract components
    not_nan, r00, r01, r02, r10, r11, r12, _, _, r22 = extract_mat_components(mat)

    # pre-initialize results
    theta_y = np.full(not_nan.shape, np.nan)
    theta_x = np.full(not_nan.shape, np.nan)
    theta_z = np.full(not_nan.shape, np.nan)

    # compute Euler angles
    theta_y[not_nan] = np.where(r12 < 1, np.where(r12 > -1, np.arctan2(r02, r22), -np.arctan2(-r01, r00)),
                                np.arctan2(-r01, r00))
    theta_x[not_nan] = np.where(r12 < 1, np.where(r12 > -1, np.arcsin(-r12), np.pi/2), -np.pi/2)
    theta_z[not_nan] = np.where(r12 < 1, np.where(r12 > -1, np.arctan2(r10, r11), 0), 0)

    return np.stack((theta_y, theta_x, theta_z), -1)


def yxy_intrinsic(mat: np.ndarray) -> np.ndarray:
    """Return yxy intrinsic Euler angle decomposition of mat (.., 4, 4))"""
    # extract components
    not_nan, r00, r01, r02, r10, r11, r12, _, r21, _ = extract_mat_components(mat)

    # pre-initialize results
    theta_y0 = np.full(not_nan.shape, np.nan)
    theta_x = np.full(not_nan.shape, np.nan)
    theta_y1 = np.full(not_nan.shape, np.nan)

    # compute Euler angles
    theta_y0[not_nan] = np.where(r11 < 1, np.where(r11 > -1, np.arctan2(-r01, -r21), 0), 0)
    theta_x[not_nan] = np.where(r11 < 1, np.where(r11 > -1, -np.arccos(r11), -np.pi), 0)
    theta_y1[not_nan] = np.where(r11 < 1, np.where(r11 > -1, np.arctan2(-r10, r12), np.arctan2(r02, r00)),
                                 np.arctan2(r02, r00))

    return np.stack((theta_y0, theta_x, theta_y1), -1)


def xzy_intrinsic(mat: np.ndarray) -> np.ndarray:
    """Return xzy intrinsic Euler angle decomposition of mat (.., 4, 4))"""
    # extract components
    not_nan, r00, r01, r02, _, r11, _, r20, r21, r22 = extract_mat_components(mat)

    # pre-initialize results
    theta_x = np.full(not_nan.shape, np.nan)
    theta_z = np.full(not_nan.shape, np.nan)
    theta_y = np.full(not_nan.shape, np.nan)

    # compute Euler angles
    theta_x[not_nan] = np.where(r01 < 1, np.where(r01 > -1, np.arctan2(r21, r11), -np.arctan2(-r20, r22)),
                                np.atan2(-r20, r22))
    theta_z[not_nan] = np.where(r01 < 1, np.where(r01 > -1, np.arcsin(-r01), np.pi/2), -np.pi/2)
    theta_y[not_nan] = np.where(r01 < 1, np.where(r01 > -1, np.arctan2(r02, r00), 0), 0)

    return np.stack((theta_x, theta_z, theta_y), -1)


def gh_isb(mat: np.ndarray) -> np.ndarray:
    """Return GH ISB (yxy intrinsic) decomposition."""
    return yxy_intrinsic(mat)


def gh_phadke(mat: np.ndarray) -> np.ndarray:
    """Return GH Phadke (xzy intrinsic) decomposition."""
    return xzy_intrinsic(mat)


def ht_isb(mat: np.ndarray) -> np.ndarray:
    """Return HT ISB (yxy intrinsic) decomposition."""
    return yxy_intrinsic(mat)


def ht_phadke(mat: np.ndarray) -> np.ndarray:
    """Return HT Phadke (xzy intrinsic) decomposition."""
    return xzy_intrinsic(mat)


def st_isb(mat: np.ndarray) -> np.ndarray:
    """Return ST ISB (yxz intrisc) decomposition."""
    return yxz_intrinsic(mat)


def thorax_isb(mat: np.ndarray) -> np.ndarray:
    """Return thorax relative to global ISB (zxy intrinsic) decomposition."""
    return zxy_intrinsic(mat)


gh_isb.legend = ['Plane of Elevation', 'Elevation(-)', 'Axial Rotation I(+)/E(-)']
ht_isb.legend = ['Plane of Elevation', 'Elevation(-)', 'Axial Rotation I(+)/E(-)']
gh_phadke.legend = ['Elevation(-)', 'Angle of Flexion(+)/Horizontal Abd(-)', 'Axial Rotation I(+)/E(-)']
ht_phadke.legend = ['Elevation(-)', 'Angle of Flexion(+)/Horizontal Abd(-)', 'Axial Rotation I(+)/E(-)']
st_isb.legend = ['Retraction(+)/Protraction(-)', 'Lateral(-)/Medial(+) Rotation', 'Anterior(-)/Posterior(+) Tilt']
thorax_isb.legend = ['Flexion(+)/Extension(-)', 'Lateral Flexion L(+)/R(-)', 'Axial Rotation L(+)/R(-)']

gh_isb.legend_short = ['PoE', 'Elevation', 'Axial']
ht_isb.legend_short = ['PoE', 'Elevation', 'Axial']
gh_phadke.legend_short = ['Elevation', 'Flex/Abd', 'Axial']
ht_phadke.legend_short = ['Elevation', 'Flex/Abd', 'Axial']
st_isb.legend_short = ['Re/Protraction', 'Rotation', 'Tilt']
thorax_isb.legend_short = ['Flex/Ext', 'L/R Flex', 'Axial']
