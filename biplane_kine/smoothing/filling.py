import numpy as np
from typing import Tuple, List
from ..misc.np_utils import find_runs
from ..kinematics.absor import absor_matrix


def fill_gaps_rb(marker_pos: np.ndarray, cluster_marker_pos: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Rigid body fill the gaps present in marker_pos using cluster_marker_pos."""
    marker_present = ~np.any(np.isnan(marker_pos), 1)
    run_values, run_starts, run_lengths = find_runs(marker_present)
    gaps_start = run_starts[~run_values]
    gaps_end = gaps_start + run_lengths[~run_values]

    marker_pos_filled = marker_pos.copy()
    gaps = list(zip(gaps_start, gaps_end))
    for gap_start, gap_end in gaps:
        assert (~np.any(np.isnan(cluster_marker_pos[:, gap_start:gap_end, :])))
        # if gap is at the beginning of the trial
        if gap_start == 0:
            marker_pos_right = marker_pos[gap_end]
            for frame_num in range(gap_start, gap_end):
                r, t = absor_matrix(cluster_marker_pos[:, gap_end, :].T, cluster_marker_pos[:, frame_num, :].T)
                marker_pos_filled[frame_num] = r.dot(marker_pos_right) + t
        # if the gap is at the end of the trial
        elif gap_end == marker_pos.shape[0]:
            marker_pos_left = marker_pos[gap_start-1]
            for frame_num in range(gap_start, gap_end):
                r, t = absor_matrix(cluster_marker_pos[:, gap_start-1, :].T, cluster_marker_pos[:, frame_num, :].T)
                marker_pos_filled[frame_num] = r.dot(marker_pos_left) + t
        # if the gap is in the middle
        else:
            marker_pos_left = marker_pos[gap_start-1]
            marker_pos_right = marker_pos[gap_end]
            for frame_num in range(gap_start, gap_end):
                total_frames = gap_end - (gap_start - 1)
                r_l, t_l = absor_matrix(cluster_marker_pos[:, gap_start-1, :].T, cluster_marker_pos[:, frame_num, :].T)
                r_r, t_r = absor_matrix(cluster_marker_pos[:, gap_end, :].T, cluster_marker_pos[:, frame_num, :].T)
                marker_pred_left = r_l.dot(marker_pos_left) + t_l
                marker_pred_right = r_r.dot(marker_pos_right) + t_r
                marker_pos_filled[frame_num] = (marker_pred_left * (gap_end-frame_num)/total_frames +
                                                marker_pred_right * (frame_num-(gap_start - 1))/total_frames)

    return marker_pos_filled, gaps
