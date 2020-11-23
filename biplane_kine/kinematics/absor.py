from typing import Tuple
import numpy as np
import quaternion


def absor_matrix(markers_a: np.ndarray, markers_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return optimal rigid transformation between markers_a (3, N) and markers_b (3, N).

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
    """Return optimal rigid transformation between markers_a (3, N) and markers_b (3, N).

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


def compute_trajectory(static_markers: np.ndarray, tracking_markers: np.ndarray) -> np.ndarray:
    """Return segment kinematic trajectory (N, 4, 4) given tracking markers in the static trial expressed in segment's
    coordinate system (K, 3) and the trajectories of the tracking markers over time in the lab
    coordinate system (K, N, 3)."""

    # markers_present (K, N)
    markers_present = ~np.any(np.isnan(tracking_markers), 2)
    num_frames = tracking_markers[0].shape[0]
    trajectory = np.full((num_frames, 4, 4), np.nan)

    for i in range(num_frames):
        # need at least 3 markers to determine the rigid transformation
        if np.count_nonzero(markers_present[:, i]) >= 3:
            # markers for this frame, removing markers with no data
            static_markers_frame = static_markers[markers_present[:, i], :]
            tracking_markers_frame = tracking_markers[markers_present[:, i], i, :]
            r, t = absor_matrix(static_markers_frame.T, tracking_markers_frame.T)
            trajectory[i, 0:3, 0:3] = r
            trajectory[i, 0:3, 3] = t
            trajectory[i, 3, :] = [0, 0, 0, 1]

    return trajectory


def compute_trajectory_continuous(static_markers: np.ndarray, tracking_markers: np.ndarray) -> np.ndarray:
    """Return segment kinematic trajectory (N, 4, 4) given tracking markers in the static trial expressed in segment's
    coordinate system (K, 3) and the trajectories of the tracking markers over time in the lab
    coordinate system (K, N, 3)."""

    # pre-initialize return value
    num_frames = tracking_markers[0].shape[0]
    trajectory = np.full((num_frames, 4, 4), np.nan)

    # remove markers that have no data at all
    no_data_markers = np.all(np.isnan(tracking_markers), (1, 2))
    data_markers = ~no_data_markers
    if np.count_nonzero(data_markers) < 3:
        return trajectory
    static_markers_wdata = static_markers[data_markers]
    tracking_markers_wdata = tracking_markers[data_markers]

    # find endpoints of cluster - do not compute kinematics prior to and post these endpoints
    cluster_present_frms = np.nonzero(~np.any(np.isnan(tracking_markers_wdata), (0, 2)))[0]

    for i in range(cluster_present_frms[0], cluster_present_frms[-1] + 1):
        r, t = absor_matrix(static_markers_wdata.T, tracking_markers_wdata[:, i, :].T)
        trajectory[i, 0:3, 0:3] = r
        trajectory[i, 0:3, 3] = t
        trajectory[i, 3, :] = [0, 0, 0, 1]

    return trajectory
