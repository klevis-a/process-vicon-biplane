import numpy as np
from scipy.signal import butter, sosfiltfilt


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
    x0_pos = pos_lowpass_filter(marker_pos_filled, start_idx, num_points=points_to_filter)
    x0_vel = np.gradient(x0_pos, dt, axis=0)
    x0_acc = np.gradient(x0_pos, dt, axis=0)
    x0 = np.array([np.mean(x0_pos[0:points_to_average], axis=0), np.mean(x0_vel[0:points_to_average], axis=0),
                   np.mean(x0_acc[0:points_to_average], axis=0)]).T
    return x0, start_idx, stop_idx
