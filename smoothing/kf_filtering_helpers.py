import numpy as np
from scipy.signal import butter, sosfiltfilt
from collections import namedtuple
from smoothing.kalman_filtering import LinearKF1DSimdKalman, LinearKF
from misc.np_utils import find_runs
import logging
log = logging.getLogger('kf_smoothing')


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
    x0 = np.array([x0_pos[0, :], np.mean(x0_vel[0:points_to_average, :], axis=0),
                   np.mean(x0_acc[0:points_to_average, :], axis=0)]).T
    return x0, start_idx, stop_idx


def post_process_raw(trial, marker_name, dt):
    marker_pos_labeled = trial.marker_data_labeled(marker_name)
    marker_pos_filled = trial.marker_data_filled(marker_name)
    # raw velocity, acceleration
    marker_vel = np.gradient(marker_pos_labeled, dt, axis=0)
    marker_acc = np.gradient(marker_vel, dt, axis=0)
    raw_means = LinearKF1DSimdKalman.StateMeans(marker_pos_labeled, marker_vel, marker_acc)

    # filled velocity, acceleration
    marker_vel_filled = np.gradient(marker_pos_filled, dt, axis=0)
    marker_acc_filled = np.gradient(marker_vel_filled, dt, axis=0)
    filled_means = LinearKF1DSimdKalman.StateMeans(marker_pos_filled, marker_vel_filled, marker_acc_filled)

    # endpoints
    raw_endpts = (0, marker_pos_labeled.shape[0])
    raw_indices = np.arange(raw_endpts[1])

    raw = FilterStep(raw_endpts, raw_indices, raw_means, None, None)
    filled = FilterStep(raw_endpts, raw_indices, filled_means, None, None)

    return raw, filled


def kf_filter_marker_piece(marker_pos_labeled, marker_pos_filled, piece_start, piece_stop, dt):
    pos_labeled_piece = marker_pos_labeled[piece_start:piece_stop, :]
    pos_filled_piece = marker_pos_filled[piece_start:piece_stop, :]
    x0, start_idx, stop_idx = x0_guess(pos_labeled_piece, pos_filled_piece, dt, 50, 10)
    p = np.tile(np.diag([1, 100, 1000])[:, :, np.newaxis], 3)
    kf = LinearKF1DSimdKalman(dt=dt, discrete_white_noise_var=10000, r=1)
    filtered_means, smoothed_means, filtered_covs, smoothed_covs = \
        kf.filter_trial_marker(pos_labeled_piece[start_idx:stop_idx, :], x0, p)

    filtered_corrs = LinearKF1DSimdKalman.CorrVec(*LinearKF1DSimdKalman.extract_corrs(filtered_covs))
    smoothed_corrs = LinearKF1DSimdKalman.CorrVec(*LinearKF1DSimdKalman.extract_corrs(smoothed_covs))

    filtered_endpts = (piece_start + start_idx, piece_start + stop_idx)
    filtered_indices = np.arange(filtered_endpts[0], filtered_endpts[1])

    filtered = FilterStep(filtered_endpts, filtered_indices, filtered_means, filtered_covs, filtered_corrs)
    smoothed = FilterStep(filtered_endpts, filtered_indices, smoothed_means, smoothed_covs, smoothed_corrs)

    return filtered, smoothed


def kf_filter_marker_all(trial, marker_name, dt):
    marker_pos_labeled = trial.marker_data_labeled(marker_name)
    marker_pos_filled = trial.marker_data_filled(marker_name)
    filtered, smoothed = kf_filter_marker_piece(marker_pos_labeled, marker_pos_filled, 0, None, dt)

    return filtered, smoothed


def kf_filter_marker_piecewise(trial, marker_name, dt, max_gap=75, min_length=75):
    marker_pos_labeled = trial.marker_data_labeled(marker_name)
    marker_pos_filled = trial.marker_data_filled(marker_name)

    start_idx, stop_idx = init_point(marker_pos_labeled, marker_pos_filled)
    runs = find_runs(~np.isnan(marker_pos_labeled[start_idx:stop_idx, 0]))
    # no data for longer than max_gap
    runs_gaps_idx = np.nonzero(((~runs[0]) & (runs[2] >= max_gap)))[0]
    num_pieces = runs_gaps_idx.size + 1
    pieces_end_idx = np.full((num_pieces, ), stop_idx)
    pieces_end_idx[:-1] = runs[1][runs_gaps_idx]
    pieces_start_idx = np.full((num_pieces,), start_idx)
    pieces_start_idx[1:] = runs[1][runs_gaps_idx + 1]

    filtered_pieces = []
    smoothed_pieces = []
    piece_number = 1
    for i in range(num_pieces):
        if (pieces_end_idx[i] - pieces_start_idx[i]) > min_length:
            log.info('Filtering piece %d running from %d to %d for trial %s marker %s.', piece_number,
                     pieces_start_idx[i], pieces_end_idx[i], trial.trial_name, marker_name)
            piece_filtered, piece_smoothed = kf_filter_marker_piece(marker_pos_labeled, marker_pos_filled,
                                                                    pieces_start_idx[i], pieces_end_idx[i], dt)
            filtered_pieces.append(piece_filtered)
            smoothed_pieces.append(piece_smoothed)
            piece_number += 1

    return filtered_pieces, smoothed_pieces


def combine_pieces(pieces):
    endpts = (pieces[0].endpts[0], pieces[-1].endpts[-1])
    indices = np.arange(*endpts)
    num_frames = endpts[-1] - endpts[0]
    pos = np.full((num_frames, 3), np.nan, dtype=np.float64)
    vel = np.full((num_frames, 3), np.nan, dtype=np.float64)
    acc = np.full((num_frames, 3), np.nan, dtype=np.float64)
    pos_cov = np.full((num_frames, 3), np.nan, dtype=np.float64)
    vel_cov = np.full((num_frames, 3), np.nan, dtype=np.float64)
    acc_cov = np.full((num_frames, 3), np.nan, dtype=np.float64)
    pos_vel_cov = np.full((num_frames, 3), np.nan, dtype=np.float64)
    pos_acc_cov = np.full((num_frames, 3), np.nan, dtype=np.float64)
    vel_acc_cov = np.full((num_frames, 3), np.nan, dtype=np.float64)
    pos_vel_corr = np.full((num_frames, 3), np.nan, dtype=np.float64)
    pos_acc_corr = np.full((num_frames, 3), np.nan, dtype=np.float64)
    vel_acc_corr = np.full((num_frames, 3), np.nan, dtype=np.float64)

    for piece in pieces:
        pos[piece.endpts[0]:piece.endpts[1], :] = piece.means.pos
        vel[piece.endpts[0]:piece.endpts[1], :] = piece.means.vel
        acc[piece.endpts[0]:piece.endpts[1], :] = piece.means.acc
        pos_cov[piece.endpts[0]:piece.endpts[1], :] = piece.covars.pos
        vel_cov[piece.endpts[0]:piece.endpts[1], :] = piece.covars.vel
        acc_cov[piece.endpts[0]:piece.endpts[1], :] = piece.covars.acc
        pos_vel_cov[piece.endpts[0]:piece.endpts[1], :] = piece.covars.pos_vel
        pos_acc_cov[piece.endpts[0]:piece.endpts[1], :] = piece.covars.pos_acc
        vel_acc_cov[piece.endpts[0]:piece.endpts[1], :] = piece.covars.vel_acc
        pos_vel_corr[piece.endpts[0]:piece.endpts[1], :] = piece.corrs.pos_vel
        pos_acc_corr[piece.endpts[0]:piece.endpts[1], :] = piece.corrs.pos_acc
        vel_acc_corr[piece.endpts[0]:piece.endpts[1], :] = piece.corrs.vel_acc

    means = LinearKF.StateMeans(pos, vel, acc)
    cov = LinearKF.CovarianceVec(pos_cov, vel_cov, acc_cov, pos_vel_cov, pos_acc_cov, vel_acc_cov)
    corr = LinearKF.CorrVec(pos_vel_corr, pos_acc_corr, vel_acc_corr)
    return FilterStep(endpts, indices, means, cov, corr)
