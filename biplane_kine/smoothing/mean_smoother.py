import numpy as np
import quaternion


def quat_mean(quats: np.ndarray) -> np.ndarray:
    """Return average of quaternions in quats (N, 4)."""
    k = np.dot(quats.T, quats)
    l, v = np.linalg.eigh(k)
    return v[:, -1]


def smooth_quat_traj(quat_traj_floats: np.ndarray, num_frames_avg: int) -> np.ndarray:
    """Return moving average filtered trajectory of quaternions in quat_traj_floats (N, 4) - scalar-first convention."""
    assert(num_frames_avg % 2 == 1)
    num_frames_ext = int((num_frames_avg - 1) / 2)
    quat_traj = quaternion.as_quat_array(quat_traj_floats)

    # extend trajectory
    quat_left_ext = quat_traj[0] * quat_traj[num_frames_ext:0:-1].conjugate() * quat_traj[0]
    quat_right_ext = quat_traj[-1] * quat_traj[-2:-2 - num_frames_ext:-1].conjugate() * quat_traj[-1]
    quat_traj_ext = np.concatenate((quat_left_ext, quat_traj, quat_right_ext))

    # compute average
    quat_traj_smooth_floats = np.zeros_like(quat_traj_floats)
    for i in range(quat_traj_smooth_floats.shape[0]):
        q_mean = quat_mean(quaternion.as_float_array(quat_traj_ext[i:i + num_frames_avg]))
        # since q and -q represent the same orientation make sure that we point in the same general direction
        quat_traj_smooth_floats[i] = -q_mean if q_mean.dot(quat_traj_floats[i]) < 0 else q_mean

    return quat_traj_smooth_floats


def smooth_pos_traj(pos_traj: np.ndarray, num_frames_avg: int) -> np.ndarray:
    """Return moving average filtered position trajectory (N, 3)."""
    assert(num_frames_avg % 2 == 1)
    num_frames_ext = int((num_frames_avg - 1) / 2)

    # extend trajectory
    pos_left_ext = -(pos_traj[num_frames_ext:0:-1] - pos_traj[0]) + pos_traj[0]
    pos_right_ext = -(pos_traj[-2:-2 - num_frames_ext:-1] - pos_traj[-1]) + pos_traj[-1]
    pos_traj_ext = np.concatenate((pos_left_ext, pos_traj, pos_right_ext))

    # average stencil
    stenc = np.ones(num_frames_avg) / num_frames_avg

    # smooth
    pos_traj_smooth = np.empty_like(pos_traj)
    for i in range(3):
        pos_traj_smooth[:, i] = np.convolve(pos_traj_ext[:, i], stenc, 'valid')

    return pos_traj_smooth
