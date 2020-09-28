from collections import namedtuple
import numpy as np
import simdkalman
from filterpy.common.discretization import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


class LinearKF:
    _CovDiagTerms = ['pos', 'vel', 'acc']
    _CovOffDiagTerms = ['pos_vel', 'pos_acc', 'vel_acc']
    _AllTerms = _CovDiagTerms + _CovOffDiagTerms
    _CovIndices = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    CovarianceVec = namedtuple('CovarianceVec', _AllTerms)
    CorrVec = namedtuple('CorrVec', _CovOffDiagTerms)
    StateMeans = namedtuple('FilterOutput', ['pos', 'vel', 'acc'])

    @classmethod
    def extract_covariances(cls, covs):
        # here we iterate over each variance/covariance pos(0), vel(1), acc(2), pos_vel(3), pos_acc(4), vel_acc(5)
        covariances = []
        for cov_idx in LinearKF._CovIndices:
            covariance = np.stack([cov[:, cov_idx[0], cov_idx[1]] for cov in covs], axis=-1)
            covariances.append(covariance)
        return covariances

    @classmethod
    def extract_corrs(cls, covariances):
        correlations = []
        for corr_name in LinearKF._CovOffDiagTerms:
            corr_name_split = corr_name.split('_')
            first_term = corr_name_split[0]
            second_term = corr_name_split[1]
            cov = getattr(covariances, corr_name)
            var_first_term = getattr(covariances, first_term)
            var_second_term = getattr(covariances, second_term)
            correlation = np.divide(cov, np.multiply(np.sqrt(var_first_term), np.sqrt(var_second_term)))
            correlations.append(correlation)
        return correlations

    @classmethod
    def extract_means(cls, means):
        # here we iterate over each derivative (0, 1, 2)
        output = []
        for i in range(3):
            mean_data = np.stack([mean[:, i] for mean in means], axis=-1)
            output.append(mean_data)
        return output


class LinearKF1DFilterPy(LinearKF):
    # this class has not gotten updated to conform to the new framework but I have left it here so the interaction with
    # FilterPy is documented somewhere
    def __init__(self, dt, discrete_white_noise_var, r, p):
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


class LinearKF1DSimdKalman(LinearKF):
    def __init__(self, dt, discrete_white_noise_var, r):
        self.dt = dt
        self.discrete_white_noise_var = discrete_white_noise_var
        self.r = r

    @classmethod
    def create_linear_kalman_filter_1d(cls, dt, discrete_white_noise_var, r):
        f = np.array([[1, dt, 0.5 * dt * dt],
                      [0, 1, dt],
                      [0, 0, 1]])
        q = Q_discrete_white_noise(dim=3, dt=dt, var=discrete_white_noise_var)
        h = np.array([[1.0, 0.0, 0.0]])
        tracker = simdkalman.KalmanFilter(state_transition=f, process_noise=q, observation_model=h, observation_noise=r)
        return tracker

    def filter_trial_marker(self, marker_data, x0, p):
        mus = []
        covs = []
        mus_smooth = []
        covs_smooth = []

        kf = LinearKF1DSimdKalman.create_linear_kalman_filter_1d(self.dt, self.discrete_white_noise_var, self.r)

        # here we iterate over each dimension x, y, z
        for n in range(3):
            result = kf.compute(marker_data[:, n], 0, initial_value=x0[n, :], initial_covariance=np.squeeze(p[:, :, n]),
                                filtered=True, smoothed=True)
            mus.append(result.filtered.states.mean)
            covs.append(result.filtered.states.cov)
            mus_smooth.append(result.smoothed.states.mean)
            covs_smooth.append(result.smoothed.states.cov)

        filtered_means = LinearKF1DSimdKalman.extract_means(mus)
        smoothed_means = LinearKF1DSimdKalman.extract_means(mus_smooth)
        filtered_covs = LinearKF1DSimdKalman.extract_covariances(covs)
        smoothed_covs = LinearKF1DSimdKalman.extract_covariances(covs_smooth)

        return (LinearKF1DSimdKalman.StateMeans(*filtered_means), LinearKF1DSimdKalman.StateMeans(*smoothed_means),
                LinearKF1DSimdKalman.CovarianceVec(*filtered_covs), LinearKF1DSimdKalman.CovarianceVec(*smoothed_covs))
