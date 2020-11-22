import numpy as np
import quaternion
from lazy import lazy
from biplane_kine.kinematics.cs import ht_r, ht_inv, change_cs
from biplane_kine.kinematics.vec_ops import extended_dot
from biplane_kine.kinematics.vel_acc import ang_vel
import biplane_kine.kinematics.euler_angles as euler_angles
from biplane_kine.smoothing.mean_smoother import smooth_pos_traj, smooth_quat_traj


class EulerTrajectory:
    """An Euler angle trajectory."""
    def __init__(self, pose_trajectory: 'PoseTrajectory'):
        self.pose_trajectory = pose_trajectory

    def __getattr__(self, decomp_method):
        ret = getattr(euler_angles, decomp_method)(self.pose_trajectory.rot_matrix)
        self.__dict__[decomp_method] = ret
        return ret


class PoseTrajectory:
    """A rigid-body pose trajectory.

    Attributes
    ----------
    pos: np.ndarray (N, 3)
        Position trajectory.
    dt: float
        Frame period - expressed in seconds.
    frame_nums: np.ndarray (N, 1)
         Frame numbers associated with trajectory.
    """

    @classmethod
    def from_quat(cls, pos, quat, dt=1, frame_nums=None):
        """Create trajectory from scalar-first quaternion (N, 4) and position (N, 3) trajectory."""
        return cls(pos=pos, quat=quat, dt=dt, frame_nums=frame_nums)

    @classmethod
    def from_matrix(cls, pos, rot_mat, dt=1, frame_nums=None):
        """Create trajectory from rotation matrix (N, 3, 3) and position (N, 3) trajectory."""
        return cls(pos=pos, rot_mat=rot_mat, dt=dt, frame_nums=frame_nums)

    @classmethod
    def from_ht(cls, ht, dt=1, frame_nums=None):
        """Create trajectory from homogeneous transformation matrix (N, 4, 4) trajectory."""
        return cls(ht=ht, dt=dt, frame_nums=frame_nums)

    def __init__(self, ht=None, pos=None, quat=None, rot_mat=None, dt=1, frame_nums=None):
        self.frame_nums = frame_nums
        self.dt = dt

        # establish position
        assert((ht is not None) ^ (pos is not None))
        if ht is not None:
            self.__dict__['ht'] = ht
            self.__dict__['rot_matrix'] = ht[..., :3, :3]
            self.pos = ht[..., :3, 3]
        else:
            self.pos = pos

        # establish orientation
        assert((quat is not None) ^ (rot_mat is not None) ^ (ht is not None))
        if quat is not None:
            assert(pos.shape[0] == quat.shape[0])
            self.__dict__['quat_float'] = quat / np.sqrt(extended_dot(quat, quat))[..., np.newaxis]
            self.__dict__['quat'] = quaternion.from_float_array(self.__dict__['quat_float'])
        elif rot_mat is not None:
            assert (pos.shape[0] == rot_mat.shape[0])
            self.__dict__['rot_matrix'] = rot_mat
        else:
            # else ht has been passed which was handled above
            pass

    @lazy
    def quat(self) -> np.ndarray:
        """Return trajectory orientation as a scalar-first quaternion (numpy.quaternion) trajectory."""
        # if self.quat has been set in constructor this will never get called
        return quaternion.from_rotation_matrix(self.rot_matrix, nonorthogonal=False)

    @property
    def quat_float(self) -> np.ndarray:
        """Return trajectory orientation as a scalar-first quaternion (numpy.float) trajectory."""
        # if self.quat_float has been set in constructor this will never get called
        return quaternion.as_float_array(self.quat)

    @lazy
    def rot_matrix(self) -> np.ndarray:
        """Return trajectory orientation as a rotation matrix (N, 3, 3) trajectory."""
        # if self.rot_matrix has been set in constructor this will never get called
        return quaternion.as_rotation_matrix(self.quat)

    @lazy
    def ht(self) -> np.ndarray:
        """Return trajectory pose as a homogeneous transformation matrix (N, 4, 4) trajectory."""
        # if self.ht has been set in constructor this will never get called
        return ht_r(self.rot_matrix, self.pos)

    def in_frame(self, ht) -> 'PoseTrajectory':
        """Return a new pose trajectory which is the current trajectory expressed in the frame defined in ht."""
        return PoseTrajectory.from_ht(change_cs(ht_inv(ht), self.ht), self.dt, self.frame_nums)

    def in_trajectory(self, traj) -> 'PoseTrajectory':
        """Return a new pose trajectory which is the current trajectory expressed in the trajectory defined in ht."""
        return PoseTrajectory.from_ht(change_cs(ht_inv(traj.ht), self.ht), self.dt, self.frame_nums)

    @lazy
    def euler(self) -> EulerTrajectory:
        """Return an EulerTrajectory object which enables various types of Euler decompositions for the current
        trajectory."""
        return EulerTrajectory(self)

    @lazy
    def vel(self) -> np.ndarray:
        """Return trajectory linear velocity."""
        return np.gradient(self.pos, self.dt, axis=0)

    @lazy
    def ang_vel(self) -> np.ndarray:
        """Return trajectory angular velocity."""
        return ang_vel(self.rot_matrix, self.dt)


def smooth_trajectory(traj: PoseTrajectory, num_frames_avg: int) -> PoseTrajectory:
    return PoseTrajectory.from_quat(smooth_pos_traj(traj.pos, num_frames_avg),
                                    smooth_quat_traj(traj.quat_float, num_frames_avg), traj.dt, traj.frame_nums)
