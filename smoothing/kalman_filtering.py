import numpy as np
from filterpy.common.discretization import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


class LinearKalmanFilter1D:
    def __init__(self, dt, discrete_white_noise_var, r, p, db):
        self.dt = dt
        self.discrete_white_noise_var = discrete_white_noise_var
        self.r = r
        self.p = p
        self.db = db

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

    def filter_trial_marker(self, marker_data):
        kfs = []
        mus = []
        covs = []
        mus_smooth = []
        covs_smooth = []

        for n in range(3):
            kf = LinearKalmanFilter1D.create_linear_kalman_filter_1d(self.dt, self.discrete_white_noise_var, self.r,
                                                                     np.array([marker_data[1, n], 0.0, 0.0]), self.p)
            kfs.append(kf)
            mu, cov, _, _ = kf.batch_filter(marker_data[:, n])
            mus.append(mu)
            covs.append(cov)
            mu_smooth, cov_smooth, _, _ = kf.rts_smoother(mu, cov)
            mus_smooth.append(mu_smooth)
            covs_smooth.append(cov_smooth)

        marker_data_filtered = np.vstack([mu[:, 0, 0] for mu in mus]).T
        marker_data_smoothed = np.vstack([mu_smooth[:, 0, 0] for mu_smooth in mus_smooth]).T

        return marker_data, marker_data_filtered, marker_data_smoothed
