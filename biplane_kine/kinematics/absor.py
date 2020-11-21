from typing import Tuple
import numpy as np
import quaternion


def absor_matrix(markers_a: np.ndarray, markers_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return optimal rigid transformation between markers_a and markers_b.

    Söderkvist, I. and P.-Å. Wedin (1993). "Determining the movements of the skeleton using well-configured markers."
    J Biomech 26(12): 1473-1477.
    """
    a_mean = np.mean(markers_a, 1)
    b_mean = np.mean(markers_b, 1)
    pos_a_centered = markers_a - a_mean[:, np.newaxis]
    pos_b_centered = markers_b - b_mean[:, np.newaxis]
    u, s, vt = np.linalg.svd(pos_b_centered @ pos_a_centered.T)
    r = u @ (np.diag([1, 1, np.linalg.det(u @ vt)])) @ vt
    t = b_mean - r.dot(a_mean)
    return r, t


def absor_quat(markers_a: np.ndarray, markers_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return optimal rigid transformation between markers_a and markers_b.

    Horn, B. K. P. (1987). "Closed-form solution of absolute orientation using unit quaternions."
    Journal of the Optical Society of America A 4(4): 629-642.
    """
    a_mean = np.mean(markers_a, 1)
    b_mean = np.mean(markers_b, 1)
    pos_a_centered = markers_a - a_mean[:, np.newaxis]
    pos_b_centered = markers_b - b_mean[:, np.newaxis]
    m = pos_a_centered @ pos_b_centered.T

    # compute n matrix detailed in paper
    delta = np.array([m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]])
    n = np.zeros((4, 4))
    n[0, :] = np.array([np.trace(m), delta[0], delta[1], delta[2]])
    n[1:4, 0] = delta
    n[1:4, 1:4] = m + m.T - np.trace(m) * np.eye(3)

    # find optimal R and t
    (eigen_values, eigen_vectors) = np.linalg.eig(n)
    q = eigen_vectors[:, eigen_values.argmax()]
    r = quaternion.as_rotation_matrix(quaternion.from_float_array(q))
    t = b_mean - r.dot(a_mean)
    return r, t
