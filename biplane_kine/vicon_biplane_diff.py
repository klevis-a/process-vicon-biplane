"""This module provides utilities for computing differences in marker position as measured by Vicon and biplane
fluoroscopy."""

from typing import Dict, Any
import numpy as np
from lazy import lazy
from biplane_kine.database.vicon_accuracy import ViconAccuracyMarkerData
from biplane_kine.kinematics.cs import vec_transform, make_vec_hom
from biplane_kine.kinematics.vec_ops import extended_dot
from biplane_kine.misc.np_utils import nanrms, nanmae
from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception


class BiplaneViconDiff:
    """Class that computes differences between marker position data recorded via Vicon and biplane fluoroscopy.

    Attributes
    ----------
    biplane_marker_data: biplane_kine.database.vicon_accuracy.ViconAccuracyMarkerData
        Marker data as measured in the biplane fluoroscopy system.
    vmd_fluoro: numpy.ndarray, (N, 3)
        Marker data as measured via Vicon transformed to the biplane fluoroscopy coordinate system.
    """
    def __init__(self, biplane_marker_data: ViconAccuracyMarkerData, vicon_endpts: np.ndarray, f_t_v: np.ndarray,
                 c3d_data_labeled: np.ndarray):
        self.biplane_marker_data = biplane_marker_data

        # transform the vicon marker data into the fluoro CS
        vicon_marker_data = c3d_data_labeled[vicon_endpts[0]:vicon_endpts[1]]
        self.vmd_fluoro = vec_transform(f_t_v, vicon_marker_data)[:, :3]

    @lazy
    def raw_diff(self) -> np.ndarray:
        """Compute vicon - biplane.

        Note that NaNs can still be present because for a particular frame a marker could be measured via biplane
        fluoroscopy but not be visible to the Vicon cameras.
        """
        return self.vmd_fluoro[self.biplane_marker_data.indices, :] - self.biplane_marker_data.data

    @lazy
    def raw_diff_scalar(self) -> np.ndarray:
        """Compute || vicon - biplane ||, where || . || indicates Euclidean norm.

        Note that NaNs can still be present because for a particular frame a marker could be measured via biplane
        fluoroscopy but not be visible to the Vicon cameras.
        """
        return np.sqrt(extended_dot(self.raw_diff, self.raw_diff))

    @lazy
    def raw_rms(self) -> np.ndarray:
        """Compute RMS(vicon - biplane)."""
        return nanrms(self.raw_diff, axis=0)

    @lazy
    def raw_rms_scalar(self) -> np.ndarray:
        """Compute RMS(|| vicon - biplane ||), where || . || indicates Euclidean norm."""
        return nanrms(self.raw_diff_scalar, axis=0)

    @lazy
    def raw_mae(self) -> np.ndarray:
        """Compute MAE(vicon - biplane), where MAE is the mean absolute error."""
        return nanmae(self.raw_diff, axis=0)

    @lazy
    def raw_mae_scalar(self) -> np.ndarray:
        """Compute MAE(|| vicon - biplane ||), where MAE is the mean absolute error and || . || indicates Euclidean
        norm."""
        return nanmae(self.raw_diff_scalar, axis=0)

    @lazy
    def raw_max(self) -> np.ndarray:
        """Compute MAX(vicon - biplane)."""
        return np.nanmax(np.absolute(self.raw_diff), axis=0)

    @lazy
    def raw_max_scalar(self) -> np.ndarray:
        """Compute MAX(|| vicon - biplane ||), where || . || indicates Euclidean norm."""
        return np.nanmax(np.absolute(self.raw_diff_scalar), axis=0)


class BiplaneViconSmoothDiff(BiplaneViconDiff):
    """Class that computes differences between marker position data recorded via Vicon then smoothed and biplane
    fluoroscopy.

    Attributes
    ----------
    use_filled_portion: bool
        Since the smoothing process also fills gaps, this attribute specifies whether to use the filled portion to
        compute statistics.
    smoothed_vmd_fluoro: numpy.ndarray, (N, 3)
        Marker data as measured via Vicon then smoothed, transformed to the biplane fluoroscopy coordinate system.
    """
    def __init__(self, biplane_marker_data: ViconAccuracyMarkerData, vicon_endpts: np.ndarray, f_t_v: np.ndarray,
                 c3d_data_labeled: np.ndarray, c3d_data_filled: np.ndarray, marker_except: Dict[str, Any], dt: float,
                 use_filled_portion=True):
        super().__init__(biplane_marker_data, vicon_endpts, f_t_v, c3d_data_labeled)
        self.use_filled_portion = use_filled_portion
        # smooth
        _, _, _, smoothed = \
            piecewise_filter_with_exception(marker_except, c3d_data_labeled, c3d_data_filled, dt)

        # now make sure that the smoothed data extends from vicon_endpts[0] to vicon_endpts[1]
        smoothed_rectified = np.full((vicon_endpts[1]-vicon_endpts[0], 3), np.nan)
        source_start_idx = vicon_endpts[0] - smoothed.endpts[0] if smoothed.endpts[0] < vicon_endpts[0] else 0
        source_stop_idx = vicon_endpts[1] - smoothed.endpts[0] if smoothed.endpts[1] > vicon_endpts[1] \
            else smoothed.endpts[1] - smoothed.endpts[0]
        target_start_idx = smoothed.endpts[0] - vicon_endpts[0] if smoothed.endpts[0] > vicon_endpts[0] else 0
        target_stop_idx = target_start_idx + (source_stop_idx - source_start_idx)
        smoothed_rectified[target_start_idx:target_stop_idx, :] = \
            smoothed.means.pos[source_start_idx:source_stop_idx, :]

        # transform the vicon marker data into the fluoro CS
        self.smoothed_vmd_fluoro = vec_transform(f_t_v, make_vec_hom(smoothed_rectified))[:, :3]

    @lazy
    def smoothed_diff(self) -> np.ndarray:
        """Compute vicon (smoothed) - biplane.

        Note that NaNs can still be present because for a particular frame a marker could be measured via biplane
        fluoroscopy but not be visible to the Vicon cameras.
        """
        if self.use_filled_portion:
            return self.smoothed_vmd_fluoro[self.biplane_marker_data.indices, :] - self.biplane_marker_data.data
        else:
            temp_diff = self.smoothed_vmd_fluoro[self.biplane_marker_data.indices, :] - self.biplane_marker_data.data
            temp_diff[np.isnan(self.vmd_fluoro[self.biplane_marker_data.indices, 0]), :] = np.nan
            return temp_diff

    @lazy
    def smoothed_diff_scalar(self) -> np.ndarray:
        """Compute || vicon (smoothed) - biplane ||, where || . || indicates Euclidean norm.

        Note that NaNs can still be present because for a particular frame a marker could be measured via biplane
        fluoroscopy but not be visible to the Vicon cameras.
        """
        return np.sqrt(extended_dot(self.smoothed_diff, self.smoothed_diff))

    @lazy
    def smoothed_rms(self) -> np.ndarray:
        """Compute RMS(vicon (smoothed) - biplane)."""
        return nanrms(self.smoothed_diff, axis=0)

    @lazy
    def smoothed_rms_scalar(self) -> np.ndarray:
        """Compute RMS(|| vicon (smoothed) - biplane ||), where || . || indicates Euclidean norm."""
        return nanrms(self.smoothed_diff_scalar, axis=0)

    @lazy
    def smoothed_mae(self) -> np.ndarray:
        """Compute MAE(vicon (smoothed) - biplane), where MAE is the mean absolute error."""
        return nanmae(self.smoothed_diff, axis=0)

    @lazy
    def smoothed_mae_scalar(self) -> np.ndarray:
        """Compute MAE(|| vicon (smoothed) - biplane ||), where MAE is the mean absolute error and || . || indicates
        Euclidean norm."""
        return nanmae(self.smoothed_diff_scalar, axis=0)

    @lazy
    def smoothed_max(self) -> np.ndarray:
        """Compute MAX(vicon (smoothed) - biplane)."""
        return np.nanmax(np.absolute(self.smoothed_diff), axis=0)

    @lazy
    def smoothed_max_scalar(self) -> np.ndarray:
        """Compute MAX(|| vicon (smoothed) - biplane ||), where || . || indicates Euclidean norm."""
        return np.nanmax(np.absolute(self.smoothed_diff_scalar), axis=0)
