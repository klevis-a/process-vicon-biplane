import numpy as np
import itertools
import distutils.util
from operator import itemgetter
from typing import Tuple, Union, List, Dict, Any
from scipy.signal import butter, sosfiltfilt
from biplane_kine.smoothing.kalman_filtering import (LinearKF1DSimdKalman, FilterStep, StateMeans, CovarianceVec,
                                                     CorrVec, extract_corrs)
from biplane_kine.misc.np_utils import find_runs
import logging
log = logging.getLogger(__name__)


class Error(Exception):
    """Base error class for this module."""
    pass


class InsufficientDataError(Error):
    """Error returned when there is insufficient data to filter/smooth."""
    pass


class DoNotUseMarkerError(Error):
    """Error returned when a marker is labeled as do not use."""
    pass


def init_point(marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray) -> Tuple[int, int]:
    """Determined endpoints [start, stop) for the marker data based on marker visibility in Vicon cameras."""
    non_nan_indices_labeled = np.nonzero(~np.isnan(marker_pos_labeled[:, 0]))[0]
    non_nan_indices_filled = np.nonzero(~np.isnan(marker_pos_filled[:, 0]))[0]
    max_non_nan = max(non_nan_indices_labeled[0], non_nan_indices_filled[0])
    # return the max of labeled and filled for starting index, but for ending ending return the last non-nan index (+1)
    # for just the labeled data
    return max_non_nan, non_nan_indices_labeled[-1] + 1


def pos_lowpass_filter(marker_pos_filled: np.ndarray, start: int, num_points: int) -> np.ndarray:
    """Low-pass filter marker position data from frame start for num_points frames."""
    # note that the filled marker data is utilized because it provides continuous data but filled marker data is not
    # utilized during the Kalman filtering process
    marker_pos_filt = np.full([num_points, 3], np.nan)
    sos = butter(4, 4, 'lowpass', fs=100, output='sos')
    for n in range(3):
        pos_filled_sub = marker_pos_filled[start:start+num_points, n]
        marker_pos_filt[:, n] = sosfiltfilt(sos, pos_filled_sub)
    return marker_pos_filt


def x0_guess(marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray, dt: float, points_to_filter: int,
             points_to_average: int, min_num_points: int = 20) -> Tuple[np.ndarray, int, int]:
    """Guess marker position, velocity, and acceleration at the beginning of the Vicon capture.

    Raises
    ------
    biplane_kine.smoothing.kalman_filtering.InsufficientDataError
    """
    start_idx, stop_idx = init_point(marker_pos_labeled, marker_pos_filled)
    if stop_idx - start_idx < min_num_points:
        raise InsufficientDataError('Not enough data to make a starting point guess')
    if start_idx + points_to_filter > stop_idx:
        points_to_filter = stop_idx - start_idx
    x0_pos = pos_lowpass_filter(marker_pos_filled, start_idx, num_points=points_to_filter)
    x0_vel = np.gradient(x0_pos, dt, axis=0)
    x0_acc = np.gradient(x0_vel, dt, axis=0)
    x0 = np.stack([x0_pos[0, :], np.mean(x0_vel[0:points_to_average, :], axis=0),
                   np.mean(x0_acc[0:points_to_average, :], axis=0)])
    return x0, start_idx, stop_idx


def post_process_raw(marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray, dt: float) \
        -> Tuple[FilterStep, FilterStep]:
    """Create FilterSteps from raw (labeled) and filled Vicon marker data."""
    # raw velocity, acceleration
    marker_vel = np.gradient(marker_pos_labeled, dt, axis=0)
    marker_acc = np.gradient(marker_vel, dt, axis=0)
    raw_means = StateMeans(marker_pos_labeled, marker_vel, marker_acc)

    # filled velocity, acceleration
    marker_vel_filled = np.gradient(marker_pos_filled, dt, axis=0)
    marker_acc_filled = np.gradient(marker_vel_filled, dt, axis=0)
    filled_means = StateMeans(marker_pos_filled, marker_vel_filled, marker_acc_filled)

    # endpoints
    raw_endpts = (0, marker_pos_labeled.shape[0])
    raw_indices = np.arange(raw_endpts[1])

    raw = FilterStep(raw_endpts, raw_indices, raw_means, None, None)
    filled = FilterStep(raw_endpts, raw_indices, filled_means, None, None)

    return raw, filled


def kf_filter_marker_piece(marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray, piece_start: int,
                           piece_stop: Union[int, None], dt: float) -> Tuple[FilterStep, FilterStep]:
    """Filter raw (labeled) Vicon marker data starting at frame piece_start and ending at frame piece_end."""
    pos_labeled_piece = marker_pos_labeled[piece_start:piece_stop, :]
    pos_filled_piece = marker_pos_filled[piece_start:piece_stop, :]
    x0, start_idx, stop_idx = x0_guess(pos_labeled_piece, pos_filled_piece, dt, 50, 10)
    # guess for initial covariance, showing increasing uncertainty for velocity and acceleration
    p = np.tile(np.diag([1, 100, 1000]), (3, 1, 1))
    kf = LinearKF1DSimdKalman(dt=dt, discrete_white_noise_var=10000, r=1)
    filtered_means, smoothed_means, filtered_covs, smoothed_covs = \
        kf.filter_marker(pos_labeled_piece[start_idx:stop_idx, :], x0, p)

    filtered_corrs = CorrVec(*extract_corrs(filtered_covs))
    smoothed_corrs = CorrVec(*extract_corrs(smoothed_covs))

    filtered_endpts = (piece_start + start_idx, piece_start + stop_idx)
    filtered_indices = np.arange(filtered_endpts[0], filtered_endpts[1])

    filtered = FilterStep(filtered_endpts, filtered_indices, filtered_means, filtered_covs, filtered_corrs)
    smoothed = FilterStep(filtered_endpts, filtered_indices, smoothed_means, smoothed_covs, smoothed_corrs)

    return filtered, smoothed


def kf_filter_marker_all(marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray, dt: float) \
        -> Tuple[FilterStep, FilterStep]:
    """Filter raw (labeled) Vicon marker data."""
    return kf_filter_marker_piece(marker_pos_labeled, marker_pos_filled, 0, None, dt)


def kf_filter_marker_piecewise(marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray, dt: float,
                               max_gap: int = 75, max_gap_secondary: Tuple[int, int] = (30, 10), min_length: int = 75) \
        -> Tuple[List[FilterStep], List[FilterStep]]:
    """Filter raw (labeled) Vicon marker data, accounting for gaps.

    There are two conditions that create a gap:
    1. The marker is not visible for more than or equal to max_gap frames
    2. Periods where marker is not visible for >= max_gap_secondary[0] frames are separated by an interval where the
    marker is visible for at most max_gap_secondary[1] frames
    Subsequently, all gaps are combined.

    Raises
    ------
    biplane_kine.smoothing.kalman_filtering.InsufficientDataError
    """
    start_idx, stop_idx = init_point(marker_pos_labeled, marker_pos_filled)
    nans_labeled = ~np.isnan(marker_pos_labeled[start_idx:stop_idx, 0])
    runs = find_runs(nans_labeled)
    # primary gaps - no data for longer than max_gap
    primary_runs_gaps_idx_start = np.nonzero(((~runs[0]) & (runs[2] >= max_gap)))[0]
    primary_runs_gaps_idx_end = primary_runs_gaps_idx_start + 1

    # secondary gaps - gaps of max_gap_secondary[0] separated by spaces where at most max_gap_secondary[1] data exists
    runs_secondary_gaps_idx = np.nonzero(((~runs[0]) & (runs[2] >= max_gap_secondary[0])))[0]
    secondary_runs_gaps_idx_start = []
    secondary_runs_gaps_idx_end = []
    for i in range(runs_secondary_gaps_idx.size-1):
        if np.sum(runs[0][runs_secondary_gaps_idx[i]+1:runs_secondary_gaps_idx[i+1]] *
                  runs[2][runs_secondary_gaps_idx[i]+1:runs_secondary_gaps_idx[i+1]]) < max_gap_secondary[1]:
            secondary_runs_gaps_idx_start.append(runs_secondary_gaps_idx[i])
            secondary_runs_gaps_idx_end.append(runs_secondary_gaps_idx[i+1]+1)

    # now let's combine the gaps
    all_runs_gaps_idx = \
        list(itertools.chain.from_iterable([zip(primary_runs_gaps_idx_start, primary_runs_gaps_idx_end),
                                            zip(secondary_runs_gaps_idx_start, secondary_runs_gaps_idx_end)]))

    def gaps_overlap(gap1: Tuple[int, int], gap2: Tuple[int, int]) -> bool:
        """Do the gaps overlap?"""
        return (gap1[0] < gap2[1]) and (gap2[0] < gap1[1])

    def combine_gaps(gap1: Tuple[int, int], gap2: Tuple[int, int]) -> Tuple[int, int]:
        """Combined the two gaps."""
        min_start = min(gap1, gap2, key=itemgetter(0))
        max_end = max(gap1, gap2, key=itemgetter(1))
        return min_start[0], max_end[1]

    # this only works if the list is sorted by the start index!
    def recursive_combine(initial_gap_list, combined_gap_list):
        # if there are no more gaps to combine then return the combined_gap_list
        if not initial_gap_list:
            return combined_gap_list

        # if we can combine the current gap (combined_gap_list[-1]) with the next gap in the list to process
        # (initial_gap_list[0])
        if gaps_overlap(combined_gap_list[-1], initial_gap_list[0]):
            # combine the gaps and update the current gap
            combined = combine_gaps(combined_gap_list[-1], initial_gap_list[0])
            combined_gap_list[-1] = combined
        else:
            # can't combine so add the considered gap becomes the current gap
            combined_gap_list.append(initial_gap_list[0])
        # either way we have taken care of this gap so remove it from the list of gaps to be considered
        del(initial_gap_list[0])
        # march forward
        return recursive_combine(initial_gap_list, combined_gap_list)

    def recursive_combine_start(initial_gap_list):
        # no gaps
        if not initial_gap_list:
            return []

        # the first combination is easy, it's just the first gap by itself
        initial_gap_list_copy = initial_gap_list.copy()
        combined_gap_list = [initial_gap_list_copy[0]]
        del(initial_gap_list_copy[0])
        return recursive_combine(initial_gap_list_copy, combined_gap_list)

    # first sort by the start index
    all_runs_gaps_idx.sort(key=itemgetter(0))
    all_runs_gaps_ids_merged = recursive_combine_start(all_runs_gaps_idx)

    # break the list of tuples apart into two lists
    runs_gaps_idx_start_final = [gap[0] for gap in all_runs_gaps_ids_merged]
    runs_gaps_idx_end_final = [gap[1] for gap in all_runs_gaps_ids_merged]

    # number of pieces to filter will always be one greater than the number of gaps
    num_pieces = len(all_runs_gaps_ids_merged) + 1
    pieces_end_idx = np.full((num_pieces, ), stop_idx)
    pieces_start_idx = np.full((num_pieces,), start_idx)
    # there may not be any gaps so check first
    if all_runs_gaps_ids_merged:
        # interior pieces run from the end index of a gap to the start index of the next gap
        pieces_end_idx[:-1] = runs[1][runs_gaps_idx_start_final] + start_idx
        pieces_start_idx[1:] = runs[1][runs_gaps_idx_end_final] + start_idx

    filtered_pieces = []
    smoothed_pieces = []
    # filter each piece
    for i in range(num_pieces):
        if (pieces_end_idx[i] - pieces_start_idx[i]) < min_length:
            log.info('Skipping Filtering piece %d running from %d to %d', i, pieces_start_idx[i], pieces_end_idx[i])
            continue

        log.info('Filtering piece %d running from %d to %d ', i, pieces_start_idx[i], pieces_end_idx[i])
        piece_filtered, piece_smoothed = kf_filter_marker_piece(marker_pos_labeled, marker_pos_filled,
                                                                pieces_start_idx[i], pieces_end_idx[i], dt)
        filtered_pieces.append(piece_filtered)
        smoothed_pieces.append(piece_smoothed)

    if not filtered_pieces:
        raise InsufficientDataError('No resulting segments to filter.')

    return filtered_pieces, smoothed_pieces


def combine_pieces(pieces: List[FilterStep]) -> FilterStep:
    """Combine multiple filtered pieces"""
    # the new endpoints runs from the start of the first piece to the end of the last piece
    endpts = (pieces[0].endpts[0], pieces[-1].endpts[1])
    indices = np.arange(*endpts)
    num_frames = endpts[1] - endpts[0]

    # preinitialize numpy containers with NaNs
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
        slc = np.s_[piece.endpts[0]-endpts[0]:piece.endpts[1]-endpts[0], :]
        pos[slc] = piece.means.pos
        vel[slc] = piece.means.vel
        acc[slc] = piece.means.acc
        pos_cov[slc] = piece.covars.pos
        vel_cov[slc] = piece.covars.vel
        acc_cov[slc] = piece.covars.acc
        pos_vel_cov[slc] = piece.covars.pos_vel
        pos_acc_cov[slc] = piece.covars.pos_acc
        vel_acc_cov[slc] = piece.covars.vel_acc
        pos_vel_corr[slc] = piece.corrs.pos_vel
        pos_acc_corr[slc] = piece.corrs.pos_acc
        vel_acc_corr[slc] = piece.corrs.vel_acc

    means = StateMeans(pos, vel, acc)
    cov = CovarianceVec(pos_cov, vel_cov, acc_cov, pos_vel_cov, pos_acc_cov, vel_acc_cov)
    corr = CorrVec(pos_vel_corr, pos_acc_corr, vel_acc_corr)
    return FilterStep(endpts, indices, means, cov, corr)


def piecewise_filter_with_exception(marker_exceptions: Dict[str, Any], marker_pos_labeled: np.ndarray,
                                    marker_pos_filled: np.ndarray, dt: float, **kwargs) \
        -> Tuple[FilterStep, FilterStep, FilterStep, FilterStep]:
    """Filter marker position data (accounting for gaps) and for exceptions specified in in marker_exceptions.

    **kwargs are passed to kf_filter_marker_piecewise once combined with marker_exceptions

    Raises
    ------
    biplane_kine.smoothing.kalman_filtering.DoNotUseMarkerError
    """
    should_use = bool(distutils.util.strtobool(marker_exceptions.get('use_marker', 'True')))
    if not should_use:
        log.warning('Skipping marker because it is labeled as DO NOT USE.')
        raise DoNotUseMarkerError('Marker has been marked as DO NOT USE')
    smoothing_params = marker_exceptions.get('smoothing_params', {})
    frame_ignores = np.asarray(marker_exceptions.get('frame_ignores', []))

    # ignore frames
    if frame_ignores.size > 0:
        marker_pos_labeled_copy = marker_pos_labeled.copy()
        marker_pos_labeled_copy[frame_ignores - 1, :] = np.nan
    else:
        marker_pos_labeled_copy = marker_pos_labeled

    combined_smoothing_params = {**kwargs, **smoothing_params}
    raw, filled = post_process_raw(marker_pos_labeled, marker_pos_filled, dt)
    filtered_pieces, smoothed_pieces = kf_filter_marker_piecewise(marker_pos_labeled_copy, marker_pos_filled, dt,
                                                                  **combined_smoothing_params)
    filtered = combine_pieces(filtered_pieces)
    smoothed = combine_pieces(smoothed_pieces)

    return raw, filled, filtered, smoothed
