import numpy as np
from .absor import absor_matrix


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
