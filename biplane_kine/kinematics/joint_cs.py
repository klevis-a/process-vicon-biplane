import numpy as np


def torso_cs_v3d(c7: np.ndarray, clav: np.ndarray, strn: np.ndarray, t10: np.ndarray, rgtr: np.ndarray,
                 lgtr: np.ndarray, lsho: np.ndarray, rsho: np.ndarray, armpit_thickness: float) -> np.ndarray:
    """Return torso frame (V3D) given skin marker positions in Vicon coordinate system."""

    def compute_distal_trunk(mid_shoulder: np.ndarray, mid_clc7: np.ndarray, mid_stt10: np.ndarray) -> np.ndarray:
        mid_shoulder_vec = mid_shoulder - mid_stt10
        mid_clc7_vec = mid_clc7 - mid_stt10
        mid_clc7_vec_unit = mid_clc7_vec / np.linalg.norm(mid_clc7_vec)
        mid_shoulder_vec_proj = mid_shoulder_vec.dot(mid_clc7_vec_unit) * mid_clc7_vec_unit
        return mid_stt10 + mid_shoulder_vec_proj

    # compute landmarks
    lsjc = lsho-np.array([0, 0, 1]) * armpit_thickness / 2
    rsjc = rsho-np.array([0, 0, 1]) * armpit_thickness / 2
    prox_trunk = (rgtr + lgtr) / 2
    mid_clc7 = (clav + c7) / 2
    mid_stt10 = (strn + t10) / 2
    mid_shoulder = (rsjc + lsjc) / 2
    distal_trunk = compute_distal_trunk(mid_shoulder, mid_clc7, mid_stt10)

    # compute coordinate system - this is translated from the Visual3D definition
    origin = distal_trunk
    y_axis = distal_trunk - prox_trunk
    # in our case this would be more aptly named thorax_z
    thorax_x_orient = np.cross(mid_stt10 - c7, clav - c7)
    thorax_x = mid_clc7 + thorax_x_orient / np.linalg.norm(thorax_x_orient) * 100
    x_axis = np.cross(y_axis, thorax_x-prox_trunk)
    z_axis = np.cross(x_axis, y_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    torso = np.eye(4)
    torso[0:3, 0] = x_axis
    torso[0:3, 1] = y_axis
    torso[0:3, 2] = z_axis
    torso[0:3, 3] = origin

    return torso


def torso_cs_isb(c7: np.ndarray, clav: np.ndarray, strn: np.ndarray, t10: np.ndarray, t5: np.ndarray) -> np.ndarray:
    """Return torso frame (ISB) given skin marker positions in Vicon coordinate system."""

    # assume that T8 is the midpoint of T10 and T5
    t8 = (t10 + t5) / 2
    strn_t8_mid = (strn + t8) / 2
    clav_c7_mid = (clav + c7) / 2
    y_axis = clav_c7_mid - strn_t8_mid
    z_axis = np.cross(clav - strn_t8_mid, c7 - clav)
    x_axis = np.cross(y_axis, z_axis)

    x_axis = x_axis/np.linalg.norm(x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)

    torso = np.eye(4)
    torso[0:3, 0] = x_axis
    torso[0:3, 1] = y_axis
    torso[0:3, 2] = z_axis
    torso[0:3, 3] = clav

    return torso


torso_cs_v3d.markers = ['C7', 'CLAV', 'STRN', 'T10', 'RGTR', 'LGTR', 'LSHO', 'RSH0']
torso_cs_isb.markers = ['C7', 'CLAV', 'STRN', 'T10', 'T5']
