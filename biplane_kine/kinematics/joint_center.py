import numpy as np

from biplane_kine.kinematics.vec_ops import extended_dot


def gamage(marker_cluster):
    """Compute joint center mean from the trajectories of the tracking markers over time in the lab coordinate system
    (K, N, 3).

    Gamage, S. S. H. U. and J. Lasenby (2002). "New least squares solutions for estimating the average centre of
    rotation and the axis of rotation." J Biomech 35(1): 87-93.
    """
    num_markers = marker_cluster.shape[0]
    num_frames = marker_cluster.shape[1]
    a = np.zeros((3, 3))
    b = np.zeros((3, 1))
    for p in range(num_markers):
        present_idx = ~np.any(np.isnan(marker_cluster[p]), axis=1)
        marker_pos = marker_cluster[p][present_idx]
        if marker_pos.size == 0:
            continue
        a_1 = (marker_pos.T @ marker_pos) / num_frames
        v_p_hat = np.mean(marker_pos, axis=0)[..., np.newaxis]
        a_2 = v_p_hat @ v_p_hat.T
        a = a + 2 * (a_1 - a_2)
        v_p_2 = extended_dot(marker_pos, marker_pos)[..., np.newaxis]
        v_p_3 = marker_pos * v_p_2
        v_p_2_hat = np.mean(v_p_2)
        v_p_3_hat = np.mean(v_p_3, axis=0)[..., np.newaxis]
        b = b + v_p_3_hat - v_p_hat * v_p_2_hat
    return np.linalg.solve(a, b)
