import numpy as np
from lazy import lazy
from biplane_kine.misc.np_utils import nanrms, nanmae
from biplane_kine.kinematics.vec_ops import extended_dot
from biplane_kine.kinematics.cs import vec_transform, make_vec_hom
from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception


class BiplaneViconDiff:
    def __init__(self, biplane_marker_data, vicon_endpts, f_t_v, c3d_data_labeled):
        self.biplane_marker_data = biplane_marker_data

        # transform the vicon marker data into the fluoro CS
        vicon_marker_data = c3d_data_labeled[vicon_endpts[0]:vicon_endpts[1]]
        self.vmd_fluoro = vec_transform(f_t_v, vicon_marker_data)[:, :3]

    @lazy
    def raw_diff(self):
        return self.vmd_fluoro[self.biplane_marker_data.indices, :] - self.biplane_marker_data.data

    @lazy
    def raw_diff_scalar(self):
        return np.sqrt(extended_dot(self.raw_diff, self.raw_diff))

    @lazy
    def raw_rms(self):
        return nanrms(self.raw_diff, axis=0)

    @lazy
    def raw_rms_scalar(self):
        return nanrms(self.raw_diff_scalar, axis=0)

    @lazy
    def raw_mae(self):
        return nanmae(self.raw_diff, axis=0)

    @lazy
    def raw_mae_scalar(self):
        return nanmae(self.raw_diff_scalar, axis=0)

    @lazy
    def raw_max(self):
        return np.nanmax(np.absolute(self.raw_diff), axis=0)

    @lazy
    def raw_max_scalar(self):
        return np.nanmax(np.absolute(self.raw_diff_scalar), axis=0)


class BiplaneViconSmoothDiff(BiplaneViconDiff):
    def __init__(self, biplane_marker_data, vicon_endpts, f_t_v, c3d_data_labeled, c3d_data_filled, marker_except, dt,
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
    def smoothed_diff(self):
        if self.use_filled_portion:
            return self.smoothed_vmd_fluoro[self.biplane_marker_data.indices, :] - self.biplane_marker_data.data
        else:
            temp_diff = self.smoothed_vmd_fluoro[self.biplane_marker_data.indices, :] - self.biplane_marker_data.data
            temp_diff[np.isnan(self.vmd_fluoro[self.biplane_marker_data.indices, 0]), :] = np.nan
            return temp_diff

    @lazy
    def smoothed_diff_scalar(self):
        return np.sqrt(extended_dot(self.smoothed_diff, self.smoothed_diff))

    @lazy
    def smoothed_rms(self):
        return nanrms(self.smoothed_diff, axis=0)

    @lazy
    def smoothed_rms_scalar(self):
        return nanrms(self.smoothed_diff_scalar, axis=0)

    @lazy
    def smoothed_mae(self):
        return nanmae(self.smoothed_diff, axis=0)

    @lazy
    def smoothed_mae_scalar(self):
        return nanmae(self.smoothed_diff_scalar, axis=0)

    @lazy
    def smoothed_max(self):
        return np.nanmax(np.absolute(self.smoothed_diff), axis=0)

    @lazy
    def smoothed_max_scalar(self):
        return np.nanmax(np.absolute(self.smoothed_diff_scalar), axis=0)
