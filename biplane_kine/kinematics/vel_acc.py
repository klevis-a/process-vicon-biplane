import numpy as np
import quaternion


def ang_vel(mat_traj: np.ndarray, dt) -> np.ndarray:
    """Return the angular velocity of the rotation matrix trajectory (N, 3, 3)."""
    mats_vel = np.gradient(mat_traj, dt, axis=0)
    mats_t = np.swapaxes(mat_traj, -2, -1)
    ang_vel_tensor = mats_vel @ mats_t
    ang_vel_vector = np.stack((ang_vel_tensor[:, 2, 1], ang_vel_tensor[:, 0, 2], ang_vel_tensor[:, 1, 0]), -1)
    return ang_vel_vector


def ang_vel_quat(quat_traj: np.ndarray, dt) -> np.ndarray:
    """Return the angular velocity of a quaternion trajectory (N, 4)."""
    mats = quaternion.as_rotation_matrix(quaternion.as_quat_array(quat_traj))
    return ang_vel(mats, dt)
