"""This module contains utilities for performing Kalman filtering/smoothing of marker data."""

from typing import NamedTuple, List, Tuple, Union
from collections.abc import Sequence
import numpy as np
import simdkalman
from filterpy.common.discretization import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


class StateMeans(NamedTuple):
    """Contains the position, velocity, and acceleration of a trajectory (whether raw, filtered, or smoothed).

    Attributes
    ----------
    pos: numpy.ndarray, (N, 3)
        Marker position.
    vel: numpy.ndarray, (N, 3)
        Marker velocity.
    acc: numpy.ndarray, (N, 3)
        Marker acceleration.
    """
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray


class CovarianceVec(NamedTuple):
    """Contains the elements of a (3x3) symmetric covariance matrix in a 6-element vector for a trajectory.

    Attributes
    ----------
    pos: numpy.ndarray, (N, 3)
        Marker position variance for each spatial dimension over the course of the trajectory.
    vel: numpy.ndarray, (N, 3)
        Marker velocity variance for each spatial dimension over the course of the trajectory.
    acc: numpy.ndarray, (N, 3)
        Marker acceleration variance for each spatial dimension over the course of the trajectory.
    pos_vel: numpy.ndarray, (N, 3)
        Marker position/velocity covariance for each spatial dimension over the course of the trajectory.
    pos_acc: numpy.ndarray, (N, 3)
        Marker position/acceleration covariance for each spatial dimension over the course of the trajectory.
    vel_acc: numpy.ndarray, (N, 3)
        Marker velocity/acceleration covariance for each spatial dimension over the course of the trajectory.
    """
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    pos_vel: np.ndarray
    pos_acc: np.ndarray
    vel_acc: np.ndarray


class CorrVec(NamedTuple):
    """Contains the correlation coefficients of position/velocity, position/acceleration, and velocity/acceleration for
    a trajectory.

    Attributes
    ----------
    pos_vel: numpy.ndarray, (N, 3)
        Marker position/velocity correlation for each spatial dimension over the course of the trajectory.
    pos_acc: numpy.ndarray, (N, 3)
        Marker position/acceleration correlation for each spatial dimension over the course of the trajectory.
    vel_acc: numpy.ndarray, (N, 3)
        Marker velocity/acceleration correlation for each spatial dimension over the course of the trajectory.
    """
    pos_vel: np.ndarray
    pos_acc: np.ndarray
    vel_acc: np.ndarray


class FilterStep(NamedTuple):
    """Contains all the data associated with a filtering or smoothing step.

    For ease and consistency, the concept of a raw step exists but covariances, and correlations do not exist.

    Attributes
    ----------
    endpts: numpy.ndarray, (2, )
        Zero-based frame indices for endpoints [start, stop) associated with this step.
    indices: numpy.ndarray, (N, )
        A convenience attribute, simply numpy.arange(endpts[0], endpts[1])
    means: numpy.ndarray, biplane_kine.smoothing.kalman_filtering.StateMeans
        The spatial position, velocity, and acceleration of the marker over the course of the trajectory
        (between endpts[0] and endpts[1]).
    covars: numpy.ndarray, biplane_kine.smoothing.kalman_filtering.CovarianceVec
        The covariance matrix (compressed into a vector) of position, velocity, and cceleration for each spatial
        dimension of the marker over the course of the trajectory (between endpts[0] and endpts[1]).
    corrs: numpy.ndarray, biplane_kine.smoothing.kalman_filtering.CorrVec
        The correlation coefficients of position, velocity, and cceleration for each spatial
        dimension of the marker over the course of the trajectory (between endpts[0] and endpts[1]).
    """
    endpts: Union[np.ndarray, Sequence]
    indices: np.ndarray
    means: StateMeans
    covars: Union[CovarianceVec, None]
    corrs: Union[CorrVec, None]


CovIndices = {
    'pos': (0, 0),
    'vel': (1, 1),
    'acc': (2, 2),
    'pos_vel': (0, 1),
    'pos_acc': (0, 2),
    'vel_acc': (1, 2),
}
"""Map from field names of CovarianceVec to corresponding indices in the covariance matrix."""


def extract_covariances(covs: List[np.ndarray]) -> List[np.ndarray]:
    """Extract covariances from covs [(n, 3, 3)_x, (n, 3, 3)_y, (n, 3, 3)_z] and return
    [(n, 3)_pos, (n, 3)_vel, (n, 3)_acc, (n, 3)_pos_vel, (n, 3)_pos_acc, (n, 3)_vel_acc]"""
    covariances = []
    for cov_idx in CovIndices.values():
        covariance = np.stack([cov[:, cov_idx[0], cov_idx[1]] for cov in covs], axis=-1)
        covariances.append(covariance)
    return covariances


def extract_corrs(covariances: CovarianceVec) -> List[np.ndarray]:
    """Extract correlations from covariances [(n, 3)_pos, (n, 3)_vel, (n, 3)_acc, (n, 3)_pos_vel, (n, 3)_pos_acc,
    (n, 3)_vel_acc] and return [(n, 3)_pos_vel, (n, 3)_pos_acc, (n, 3)_vel_acc]"""
    correlations = []
    for corr_name in CorrVec._fields:
        # split something like pos_vel into pos and vel
        corr_name_split = corr_name.split('_')
        first_term = corr_name_split[0]
        second_term = corr_name_split[1]
        # get covariance of pos_vel
        cov = getattr(covariances, corr_name)
        # get variance of pos
        var_first_term = getattr(covariances, first_term)
        # get variance of vel
        var_second_term = getattr(covariances, second_term)
        # compute correlation
        correlation = np.divide(cov, np.multiply(np.sqrt(var_first_term), np.sqrt(var_second_term)))
        correlations.append(correlation)
    return correlations


def extract_means(means: List[np.ndarray]) -> np.ndarray:
    """Extract means from means [(n, 3)_x (n, 3)_y (n, 3)_z] and return [(n, 3)_pos (n, 3)_vel (n, 3)_acc]"""

    # 1. start with means which is a list of (N, 3) numpy arrays, where the last dimension represents kinematic variable
    #    (pos, vel, acc)
    # 2. np.stack creates a (N, 3, 3) array of frame index x kinematic variable x spatial dimension (x, y, z)
    # 3. swapaxes creates a view of the above array as (3, N, 3) - kinematic variable x frame index x spatial dimension
    # 4. return a copy since we want the optimized memory layout of the array as in Step 3
    return np.swapaxes(np.stack(means, axis=-1), 0, 1).copy()


class LinearKF1DFilterPy:
    # This class is not used and has not gotten updated to conform to the new framework but I have left it here so the
    # interaction with FilterPy is documented somewhere
    def __init__(self, dt, discrete_white_noise_var, r, p):
        self.dt = dt
        self.discrete_white_noise_var = discrete_white_noise_var
        self.r = r
        self.p = p

    @staticmethod
    def create_linear_kalman_filter_1d(dt, discrete_white_noise_var, r, x, p):
        f = np.array([[1, dt, 0.5 * dt * dt],
                      [0, 1, dt],
                      [0, 0, 1]])
        q = Q_discrete_white_noise(dim=3, dt=dt, var=discrete_white_noise_var)
        tracker = KalmanFilter(dim_x=3, dim_z=1)
        tracker.F = f
        tracker.Q = q
        tracker.x = np.array([x]).T
        tracker.P = p
        tracker.R = np.eye(1) * r
        tracker.H = np.array([[1.0, 0.0, 0.0]])
        return tracker

    def filter_trial_marker(self, marker_data_orig):
        # noinspection PyTypeChecker
        marker_data = np.where(np.isnan(marker_data_orig), None, marker_data_orig)
        mus = []
        covs = []
        mus_smooth = []
        covs_smooth = []

        # here we iterate over each dimension x, y, z
        for n in range(3):
            kf = LinearKF1DFilterPy.create_linear_kalman_filter_1d(self.dt, self.discrete_white_noise_var, self.r,
                                                                   np.array([marker_data[0, n], 0.0, 0.0]), self.p)
            mu, cov, _, _ = kf.batch_filter(marker_data[:, n])
            mus.append(mu)
            covs.append(cov)
            mu_smooth, cov_smooth, _, _ = kf.rts_smoother(mu, cov)
            mus_smooth.append(mu_smooth)
            covs_smooth.append(cov_smooth)

        # here we iterate over each derivative (0, 1, 2)
        filtered_outputs = []
        smoothed_outputs = []
        for i in range(3):
            marker_data_filtered = np.vstack([mu[:, i, 0] for mu in mus]).T
            marker_data_smoothed = np.vstack([mu_smooth[:, i, 0] for mu_smooth in mus_smooth]).T
            filtered_outputs.append(marker_data_filtered)
            smoothed_outputs.append(marker_data_smoothed)

        return filtered_outputs, smoothed_outputs, covs, covs_smooth


class LinearKF1DSimdKalman:
    """Linear Kalman filter based on simdkalman library.

    Assumes a piecewise discrete white noise for the process model. The state variables for the model are position,
    velocity, and acceleration. The measured variables are comprised only of position. In practice
    'discrete_white_noise_var' is determined empirically, although a good rule of thumb is to use an initial guess
    somewhere between 1/2 da and da, where da is the maximum predicted change in acceleration between time periods.

    Attributes
    ----------
    dt: float
        Time, in seconds, between each measurement.
    discrete_white_noise_var: float
        Acceleration variance used to compute the process model noise (e.g. mm^2/s^4).
    r: float
        Observation noise, measured in terms of variance (e.g. mm^2)
    """
    def __init__(self, dt: float, discrete_white_noise_var: float, r: float):
        self.dt = dt
        self.discrete_white_noise_var = discrete_white_noise_var
        self.r = r
        self.kf = LinearKF1DSimdKalman.create_linear_kalman_filter_1d(self.dt, self.discrete_white_noise_var, self.r)

    @staticmethod
    def create_linear_kalman_filter_1d(dt: float, discrete_white_noise_var: float, r: float) -> simdkalman.KalmanFilter:
        """Create a simdkalman filter."""
        # dynamic (process) model - 1D equation of motion
        f = np.array([[1, dt, 0.5 * dt * dt],
                      [0, 1, dt],
                      [0, 0, 1]])
        # assume piecewise discrete white noise for the process (dynamic) model
        q = Q_discrete_white_noise(dim=3, dt=dt, var=discrete_white_noise_var)
        # measurement model - we only know position
        h = np.array([[1.0, 0.0, 0.0]])
        return simdkalman.KalmanFilter(state_transition=f, process_noise=q, observation_model=h, observation_noise=r)

    def filter_marker(self, marker_data: np.ndarray, x0: np.ndarray, p: np.ndarray) \
            -> Tuple[StateMeans, StateMeans, CovarianceVec, CovarianceVec]:
        """Filter marker data, numpy.ndarray (N, 3) given an initial guess of marker position (x0) with covariance of
        p."""
        mus = []
        covs = []
        mus_smooth = []
        covs_smooth = []

        # here we iterate over each dimension x, y, z
        for n in range(3):
            result = self.kf.compute(marker_data[:, n], 0, initial_value=x0[n, :],
                                     initial_covariance=np.squeeze(p[:, :, n]), filtered=True, smoothed=True)
            mus.append(result.filtered.states.mean)
            covs.append(result.filtered.states.cov)
            mus_smooth.append(result.smoothed.states.mean)
            covs_smooth.append(result.smoothed.states.cov)

        filtered_means = extract_means(mus)
        smoothed_means = extract_means(mus_smooth)
        filtered_covs = extract_covariances(covs)
        smoothed_covs = extract_covariances(covs_smooth)

        return (StateMeans(*filtered_means), StateMeans(*smoothed_means), CovarianceVec(*filtered_covs),
                CovarianceVec(*smoothed_covs))
