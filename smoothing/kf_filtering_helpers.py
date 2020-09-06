import numpy as np
from scipy.signal import butter, sosfiltfilt
from collections import namedtuple
from smoothing.kalman_filtering import LinearKF1DSimdKalman

FilterStep = namedtuple('FilterStep', ['endpts', 'indices', 'means', 'covars', 'corrs'])


class InsufficientDataError(Exception):
    pass


def init_point(marker_pos_labeled, marker_pos_filled):
    non_nan_indices_labeled = np.nonzero(~np.isnan(marker_pos_labeled[:, 0]))[0]
    non_nan_indices_filled = np.nonzero(~np.isnan(marker_pos_filled[:, 0]))[0]
    max_non_nan = max(non_nan_indices_labeled[0], non_nan_indices_filled[0])
    # return the max of labeled and filled for starting index, but for ending ending return the last non-nan index (+1)
    # for just the labeled data
    return max_non_nan, non_nan_indices_labeled[-1] + 1


def pos_lowpass_filter(marker_pos_filled, start, num_points):
    # note that the filled marker data is utilized because it provides continuous data but filled marker data is not
    # utilized during the Kalman filtering process
    marker_pos_filt = np.full([num_points, 3], np.nan)
    sos = butter(4, 4, 'lowpass', fs=100, output='sos')
    for n in range(3):
        pos_filled_sub = marker_pos_filled[start:start+num_points, n]
        marker_pos_filt[:, n] = sosfiltfilt(sos, pos_filled_sub)
    return marker_pos_filt


def x0_guess(marker_pos_labeled, marker_pos_filled, dt, points_to_filter, points_to_average):
    start_idx, stop_idx = init_point(marker_pos_labeled, marker_pos_filled)
    if stop_idx - start_idx < 20:
        raise InsufficientDataError
    if start_idx + points_to_filter > stop_idx:
        points_to_filter = stop_idx - start_idx
    x0_pos = pos_lowpass_filter(marker_pos_filled, start_idx, num_points=points_to_filter)
    x0_vel = np.gradient(x0_pos, dt, axis=0)
    x0_acc = np.gradient(x0_vel, dt, axis=0)
    x0 = np.array([x0_pos[0], np.mean(x0_vel[0:points_to_average], axis=0),
                   np.mean(x0_acc[0:points_to_average], axis=0)]).T
    return x0, start_idx, stop_idx


def kf_filter_marker(trial, marker_name, dt):
    marker_pos_labeled = trial.marker_data_labeled(marker_name)
    marker_pos_filled = trial.marker_data_filled(marker_name)
    x0, start_idx, stop_idx = x0_guess(marker_pos_labeled, marker_pos_filled, dt, 50, 10)
    p = np.tile(np.diag([1, 100, 1000])[:, :, np.newaxis], 3)
    kf = LinearKF1DSimdKalman(dt=dt, discrete_white_noise_var=10000, r=1)
    filtered_means, smoothed_means, filtered_covs, smoothed_covs = \
        kf.filter_trial_marker(marker_pos_labeled[start_idx:stop_idx, :], x0, p)

    # post process
    marker_vel = np.gradient(marker_pos_labeled, dt, axis=0)
    marker_acc = np.gradient(marker_vel, dt, axis=0)
    raw_means = LinearKF1DSimdKalman.StateMeans(marker_pos_labeled, marker_vel, marker_acc)
    marker_vel_filled = np.gradient(marker_pos_filled, dt, axis=0)
    marker_acc_filled = np.gradient(marker_vel_filled, dt, axis=0)
    filled_means = LinearKF1DSimdKalman.StateMeans(marker_pos_filled, marker_vel_filled, marker_acc_filled)
    filtered_corrs = LinearKF1DSimdKalman.CorrVec(*LinearKF1DSimdKalman.extract_corrs(filtered_covs))
    smoothed_corrs = LinearKF1DSimdKalman.CorrVec(*LinearKF1DSimdKalman.extract_corrs(smoothed_covs))

    raw_endpts = (0, marker_pos_labeled.shape[0])
    raw_indices = np.arange(raw_endpts[1])
    filtered_endpts = (start_idx, stop_idx)
    filtered_indices = np.arange(filtered_endpts[0], filtered_endpts[1])

    raw = FilterStep(raw_endpts, raw_indices, raw_means, None, None)
    filled = FilterStep(raw_endpts, raw_indices, filled_means, None, None)
    filtered = FilterStep(filtered_endpts, filtered_indices, filtered_means, filtered_covs, filtered_corrs)
    smoothed = FilterStep(filtered_endpts, filtered_indices, smoothed_means, smoothed_covs, smoothed_corrs)

    return raw, filled, filtered, smoothed
