from collections import namedtuple
import numpy as np
import simdkalman
from filterpy.common.discretization import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

FilterOutput = namedtuple('FilterOutput', ['pos', 'vel', 'acc'])


class LinearKF1DFilterPy:
    def __init__(self, dt, discrete_white_noise_var, r, p):
        print('Using: %s' % self.__class__.__name__)
        self.dt = dt
        self.discrete_white_noise_var = discrete_white_noise_var
        self.r = r
        self.p = p

    @classmethod
    def create_linear_kalman_filter_1d(cls, dt, discrete_white_noise_var, r, x, p):
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
                                                                   np.array([marker_data[1, n], 0.0, 0.0]), self.p)
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

        return FilterOutput._make(filtered_outputs), FilterOutput._make(smoothed_outputs), covs, covs_smooth


class LinearKF1DSimdKalman:
    def __init__(self, dt, discrete_white_noise_var, r, p):
        print('Using: %s' % self.__class__.__name__)
        self.dt = dt
        self.discrete_white_noise_var = discrete_white_noise_var
        self.r = r
        self.p = p

    @classmethod
    def create_linear_kalman_filter_1d(cls, dt, discrete_white_noise_var, r):
        f = np.array([[1, dt, 0.5 * dt * dt],
                      [0, 1, dt],
                      [0, 0, 1]])
        q = Q_discrete_white_noise(dim=3, dt=dt, var=discrete_white_noise_var)
        h = np.array([[1.0, 0.0, 0.0]])
        tracker = simdkalman.KalmanFilter(state_transition=f, process_noise=q, observation_model=h, observation_noise=r)
        return tracker

    def filter_trial_marker(self, marker_data_orig):
        marker_data = marker_data_orig
        mus = []
        covs = []
        mus_smooth = []
        covs_smooth = []

        kf = LinearKF1DSimdKalman.create_linear_kalman_filter_1d(self.dt, self.discrete_white_noise_var, self.r)
        # here we iterate over each dimension x, y, z
        for n in range(3):
            first_non_nan = marker_data[~np.isnan(marker_data[:, n]), n][0]
            result = kf.compute(marker_data[:, n], 0, initial_value=np.array([first_non_nan, 0.0, 0.0]),
                                initial_covariance=self.p, smoothed=True, filtered=True)
            mus.append(result.filtered.states.mean)
            covs.append(result.filtered.states.cov)
            mus_smooth.append(result.smoothed.states.mean)
            covs_smooth.append(result.smoothed.states.cov)

        # here we iterate over each derivative (0, 1, 2)
        filtered_outputs = []
        smoothed_outputs = []
        for i in range(3):
            marker_data_filtered = np.vstack([mu[:, i] for mu in mus]).T
            marker_data_smoothed = np.vstack([mu_smooth[:, i] for mu_smooth in mus_smooth]).T
            filtered_outputs.append(marker_data_filtered)
            smoothed_outputs.append(marker_data_smoothed)

        return FilterOutput._make(filtered_outputs), FilterOutput._make(smoothed_outputs), covs, covs_smooth
