"""A module that provides plotters for marker data."""

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from typing import Union, Sequence, List
from biplane_kine.graphing.smoothing_graph_utils \
    import (marker_graph_init, marker_graph_add, marker_graph_title, add_vicon_start_stop, marker_diff_his_init,
            marker_diff_his_add, marker_graph_add_cov, cov_trend_graph_init, cov_trend_graph_add)
from biplane_kine.graphing.common_graph_utils import make_interactive
from biplane_kine.smoothing.kalman_filtering import FilterStep


class LabeledMarkerPloter:
    """Simple plotter for Vicon marker data.

    Attributes
    ----------
    trial_name : str
        The Vicon trial identifier.
    marker_name : str
        The marker identifier.
    marker_pos_labeled: numpy.ndarray, (N, 3)
        The labeled marker position data, (N, 3).
    vicon_endpts: array-like
        The frame indices (endpoints) of the Vicon trial that correspond to the endpoints of the reciprocal
        biplane fluoroscopy trial.
    """

    def __init__(self, trial_name: str, marker_name: str, marker_pos_labeled: np.ndarray,
                 vicon_endpts: Union[np.ndarray, Sequence]):
        self.trial_name = trial_name
        self.marker_name = marker_name
        self.marker_pos_labeled = marker_pos_labeled
        self.vicon_endpts = vicon_endpts
        self.frame_nums = np.arange(self.marker_pos_labeled.shape[0]) + 1

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot marker_data onto 3 row subplots on Figure 0."""
        fig = plt.figure(num=0)
        ax = fig.subplots(3, 1, sharex=True)
        marker_graph_init(ax, self.marker_pos_labeled, 'Pos (mm)', x_data=self.frame_nums, color='blue')
        add_vicon_start_stop(ax, self.vicon_endpts[0] + 1, self.vicon_endpts[1])
        marker_graph_title(fig, self.trial_name + ' ' + self.marker_name)
        make_interactive()
        return [fig]


class LabeledFilledMarkerPlotter(LabeledMarkerPloter):
    """Plotter that demonstrates differences between labeled and filled for marker data.

    Attributes
    ----------
    marker_pos_filled: numpy.ndarray, (N, 3)
        The labeled marker position data.
    """

    def __init__(self, trial_name: str, marker_name: str, marker_pos_labeled: np.ndarray, marker_pos_filled: np.ndarray,
                 vicon_endpts: Union[np.ndarray, Sequence]):
        super().__init__(trial_name, marker_name, marker_pos_labeled, vicon_endpts)
        self.marker_pos_filled = marker_pos_filled

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot overlayed labeled and filled marker_data onto 3 row subplots on Figure 0."""
        fig = plt.figure(num=0)
        ax = fig.subplots(3, 1, sharex=True)
        marker_graph_init(ax, self.marker_pos_filled, 'Pos (mm)', x_data=self.frame_nums, color='red')
        marker_graph_add(ax, self.marker_pos_labeled, self.frame_nums, color='indigo', marker='.')
        add_vicon_start_stop(ax, self.vicon_endpts[0] + 1, self.vicon_endpts[1])
        marker_graph_title(fig, self.trial_name + ' ' + self.marker_name)
        make_interactive()
        return [fig]


class SmoothingDebugPlotter:
    """Plotter that enables debugging Kalman smoothing of marker data.

    Attributes
    ----------
    trial_name: str
        Trial identifier.
    marker_name: str
        Marker identifier.
    raw: biplane_kine.smoothing.kalman_filtering.FilterStep
        Raw Vicon marker position data.
    filtered: biplane_kine.smoothing.kalman_filtering.FilterStep
        Filtered Vicon marker position data.
    smoothed: biplane_kine.smoothing.kalman_filtering.FilterStep
        Smoothed Vicon marker position data.
    frames: numpy.ndarray, (N,)
        Frame numbers (1-based indexing) corresponding to raw Vicon marker data.
    filtered_frames: numpy.ndarray, (N,)
        Frame numbers (1-based indexing) corresponding to filtered/smoothed Vicon marker data.
    vicon_endpts: numpy.ndarray, (2,)
        Zero-based indices (endpoints) that indicate which of the frames in the Vicon trial correspond to the endpoints
        of the reciprocal biplane fluoroscopy trial.
    vicon_frame_endpts: numpy.ndarray, (2,)
        One-based frame indices (endpoints) that indicate which of the frames in the Vicon trial correspond to the
        endpoints of the reciprocal biplane fluoroscopy trial.
    filtered_pos_diff: numpy.ndarray, (N, 3)
        Difference between filtered and raw Vicon marker position data.
    smoothed_pos_diff: numpy.ndarray, (N, 3)
        Difference between smoothed and raw Vicon marker position data.
    """

    def __init__(self, trial_name: str, marker_name: str, raw: FilterStep, filtered: FilterStep, smoothed: FilterStep,
                 vicon_endpts: Union[np.ndarray, Sequence]):
        self.trial_name = trial_name
        self.marker_name = marker_name
        self.raw = raw
        self.filtered = filtered
        self.smoothed = smoothed
        self.frames = self.raw.indices + 1
        self.filtered_frames = self.filtered.indices + 1
        self.vicon_endpts = vicon_endpts
        self.vicon_frame_endpts = [vicon_endpts[0] + 1, vicon_endpts[1]]
        self.filtered_pos_diff = self.filtered.means.pos - self.raw.means.pos[self.filtered.endpts[0]:
                                                                              self.filtered.endpts[-1], :]
        self.smoothed_pos_diff = self.smoothed.means.pos - self.raw.means.pos[self.filtered.endpts[0]:
                                                                              self.filtered.endpts[-1], :]

    def plot(self, plot_diff: bool = True) -> List[matplotlib.figure.Figure]:
        """Plot figures that allow debugging of smoothed marker data.

        Figure 0: Raw, filtered, smoothed position trend plots covering the biplane fluoroscopy capture interval.
        Figure 1: Raw, filtered, smoothed position trend plots covering the Vicon capture interval.
        Figure 2: Raw, filtered, smoothed velocity trend plots covering the Vicon capture interval.
        Figure 3: Raw, filtered, smoothed acceleration trend plots covering the Vicon capture interval.
        Figure 4: Smoothed velocity trend plots covering the biplane Vicon interval.
        Figure 5: Smoothed acceleration trend plots covering the Vicon capture interval.
        Figure 6: Trend plots of the difference between filtered/smoothed marker data and raw marker data,
                  covering the Vicon capture interval.
        Figure 7: Histogram of the difference between filtered/smoothed marker data and raw marker data,
                  covering the biplane fluoroscopy capture interval.
        Figure 8: Trend plots of position, velocity, and acceleration variance for filter/smoothed marker data,
                  covering the Vicon capture interval.
        Figure 9: Trend plots of position/velocity, position/acceleration, velocity/acceleration correlation for
                  filter/smoothed marker data, covering the Vicon capture interval.
        """
        y_labels = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s$^2$)']
        y_labels_corr = ['Pos Vel Corr', 'Pos Acc Corr', 'Vel Acc Corr']
        attrs = ['pos', 'vel', 'acc']
        title = self.trial_name + ' ' + self.marker_name

        current_fig_num = 0
        figs = []
        pos_fig = self.plot_marker_data(title, y_labels[0], 'pos', current_fig_num, add_sd=False, clip_graph=True,
                                        marker='.')
        current_fig_num += 1
        figs.append(pos_fig)

        for (y_label, attr) in zip(y_labels, attrs):
            if attr == 'pos':
                kine_fig = self.plot_marker_data(title, y_label, attr, current_fig_num, marker='.')
            else:
                kine_fig = self.plot_marker_data(title, y_label, attr, current_fig_num, marker='')
            current_fig_num += 1
            figs.append(kine_fig)

        smooth_vel = self.plot_marker_data_smooth(title, y_labels[1], 'vel', current_fig_num)
        current_fig_num += 1
        figs.append(smooth_vel)
        smooth_acc = self.plot_marker_data_smooth(title, y_labels[2], 'acc', current_fig_num)
        current_fig_num += 1
        figs.append(smooth_acc)

        if plot_diff:
            trend_diff = self.plot_marker_data_diff(title, 'Filtering Effect (mm)', current_fig_num)
            current_fig_num += 1
            figs.append(trend_diff)
            hist_diff = self.plot_marker_data_diff_hist(title, 'Filtering Effect (mm)', current_fig_num)
            current_fig_num += 1
            figs.append(hist_diff)

        var_plot = self.plot_cov(title, y_labels, current_fig_num)
        current_fig_num += 1
        figs.append(var_plot)
        corr_plot = self.plot_corr(title, y_labels_corr, current_fig_num)
        current_fig_num += 1
        figs.append(corr_plot)

        return figs

    def plot_marker_data(self, title: str, y_label: str, kine_var: str, fig_num: int, add_sd: bool = True,
                         clip_graph: bool = False, marker: str = '') -> matplotlib.figure.Figure:
        """Plot raw, filtered, and smoothed kinematic (position, velocity, acceleration) marker data, allowing the
        addition of confidence bounds."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_raw = marker_graph_init(ax, getattr(self.raw.means, kine_var), y_label, x_data=self.frames,
                                      color='indigo', marker=marker)
        lines_filtered = marker_graph_add(ax, getattr(self.filtered.means, kine_var), self.filtered_frames, color='red')
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames,
                                          color='limegreen')

        if add_sd:
            marker_graph_add_cov(ax, getattr(self.filtered.means, kine_var), getattr(self.filtered.covars, kine_var),
                                 self.filtered_frames, 'red')
            marker_graph_add_cov(ax, getattr(self.smoothed.means, kine_var), getattr(self.smoothed.covars, kine_var),
                                 self.filtered_frames, 'limegreen')
        if clip_graph:
            for c_ax in ax:
                c_ax.set_xlim(self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        else:
            add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        fig.legend((lines_raw[0], lines_filtered[0], lines_smoothed[0]), ('Raw', 'Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1)
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_marker_data_smooth(self, title: str, y_label: str, kine_var: str, fig_num: int) \
            -> matplotlib.figure.Figure:
        """Plot smoothed kinematic (position, velocity, acceleration) marker data."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_smoothed = marker_graph_init(ax, getattr(self.smoothed.means, kine_var), y_label,
                                           x_data=self.filtered_frames, color='limegreen')
        fig.legend([lines_smoothed[0]], ['Smoothed'], 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_marker_data_diff(self, title: str, y_label: str, fig_num: int) -> matplotlib.figure.Figure:
        """Plot (filtered and smoothed marker positions) - (raw marker positions)."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_filtered = marker_graph_init(ax, self.filtered_pos_diff, y_label, x_data=self.filtered_frames,
                                           color='red')
        lines_smoothed = marker_graph_add(ax, self.smoothed_pos_diff, self.filtered_frames, color='limegreen')
        fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_marker_data_diff_hist(self, title: str, x_label: str, fig_num: int, clip_graph: bool = True) \
            -> matplotlib.figure.Figure:
        """Plot histogram of (filtered and smoothed marker positions) - (raw marker positions)."""
        if clip_graph:
            endpts = np.zeros((2,), dtype=np.int)

            if self.vicon_endpts[0] > self.filtered.endpts[0]:
                endpts[0] = self.vicon_endpts[0] - self.filtered.endpts[0]

            if self.vicon_endpts[1] < self.filtered.endpts[1]:
                endpts[1] = self.vicon_endpts[1] - self.filtered.endpts[0]
            else:
                endpts[1] = self.filtered.endpts[1] - self.filtered.endpts[0]

            filtered_diff = self.filtered_pos_diff[endpts[0]:endpts[1]]
            smoothed_diff = self.smoothed_pos_diff[endpts[0]:endpts[1]]
        else:
            filtered_diff = self.filtered_pos_diff
            smoothed_diff = self.smoothed_pos_diff

        fig = plt.figure(num=fig_num)
        ax = fig.subplots(1, 3, sharey=True)
        lines_filtered = marker_diff_his_init(ax, filtered_diff, x_label, 'red')
        lines_smoothed = marker_diff_his_add(ax, smoothed_diff, 'limegreen')
        fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_cov(self, title: str, y_labels: Sequence[str], fig_num: int) -> matplotlib.figure.Figure:
        """Plot overlayed variance of filtered and smoothed kinematic variables (position, velocity, acceleration) in
        separate rows with 3 columns for each spatial dimension (3x3). """
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 3, sharex='all', sharey='row')
        lines_filtered = cov_trend_graph_init(ax, self.filtered.covars, self.filtered_frames, y_labels, np.sqrt,
                                              color='red')
        lines_smooth = cov_trend_graph_add(ax, self.smoothed.covars, self.filtered_frames, np.sqrt, color='limegreen')
        fig.legend((lines_filtered[0][0], lines_smooth[0][0]), ('Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=2)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_corr(self, title: str, y_labels: Sequence[str], fig_num: int) -> matplotlib.figure.Figure:
        """Plot overlayed correlation of filtered and smoothed kinematic variables (position/velocity,
        position/acceleration, velocity/acceleration) in separate rows with 3 columns for each spatial
        dimension (3x3). """
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 3, sharex='all', sharey='row')
        lines_filtered = cov_trend_graph_init(ax, self.filtered.corrs, self.filtered_frames, y_labels, lambda x: x,
                                              color='red')
        lines_smooth = cov_trend_graph_add(ax, self.smoothed.corrs, self.filtered_frames, lambda x: x,
                                           color='limegreen')
        fig.legend((lines_filtered[0][0], lines_smooth[0][0]), ('Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=2)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        marker_graph_title(fig, title)
        make_interactive()
        return fig


class SmoothingOutputPlotter(SmoothingDebugPlotter):
    """Plotter for outputing a PDF record of the Kalman smoothing of marker data.
    
    Attributes
    ---------
    filled: biplane_kine.smoothing.kalman_filtering.FilterStep
        Filled Vicon marker position data.
    """
    def __init__(self, trial_name: str, marker_name: str, raw: FilterStep, filled: FilterStep, filtered: FilterStep,
                 smoothed: FilterStep, vicon_endpts: Union[np.ndarray, Sequence]):
        super().__init__(trial_name, marker_name, raw, filtered, smoothed, vicon_endpts)
        self.filled = filled

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot figures that allow recording performance of Kalman smoother for marker data.

        Figure 0: Raw, filled, smoothed position trend plots covering the biplane fluoroscopy capture interval.
        Figure 1: Raw, filled, smoothed position trend plots covering the Vicon capture interval.
        Figure 2: Raw and smoothed velocity trend plots covering the Vicon capture interval.
        Figure 3: Trend plots of difference between smoothed and raw marker positions covering the Vicon capture
                  interval.
        Figure 3: Histogram of difference between smoothed and raw marker positions covering the Vicon capture
                  interval.
        """
        y_labels = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s$^2$)']
        y_label_diff = 'Filtering Effect (mm)'
        title = self.trial_name + ' ' + self.marker_name

        figs = []
        # 0 - pos clipped
        pos_fig_clip = self.plot_pos_data(title, y_labels[0], 'pos', fig_num=0, add_sd=False, clip_graph=True)
        figs.append(pos_fig_clip)
        # 1 - pos all
        pos_fig_all = self.plot_pos_data(title, y_labels[0], 'pos', fig_num=1, add_sd=False, clip_graph=False)
        figs.append(pos_fig_all)
        # 2 - velocity
        vel_fig = self.plot_marker_data(title, y_labels[1], 'vel', 2)
        figs.append(vel_fig)
        # 3 - trend diff
        trend_diff = self.plot_marker_data_diff(title, y_label_diff, fig_num=3)
        figs.append(trend_diff)
        # 4 - hist diff
        hist_diff = self.plot_marker_data_diff_hist(title, y_label_diff, 4)
        figs.append(hist_diff)

        return figs

    def plot_pos_data(self, title: str, y_label: str, kine_var: str, fig_num: int, add_sd: bool = True,
                      clip_graph: bool = False) -> matplotlib.figure.Figure:
        """Plot overlayed raw, filled, and smoothed marker position data."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_filled = marker_graph_init(ax, self.filled.means.pos, y_label, x_data=self.frames, color='red')
        lines_raw = marker_graph_add(ax, self.raw.means.pos, self.frames, color='indigo', marker='.')
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames,
                                          color='limegreen', lw=1)

        if add_sd:
            marker_graph_add_cov(ax, self.smoothed.means.pos, self.smoothed.covars.pos, self.filtered_frames,
                                 'limegreen')
        if clip_graph:
            for c_ax in ax:
                c_ax.set_xlim(self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        else:
            add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])

        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
        fig.suptitle(title, x=0.25, fontsize=11, fontweight='bold')
        plt.subplots_adjust(top=0.94)
        fig.legend((lines_raw[0], lines_filled[0], lines_smoothed[0]), ('Raw', 'Filled', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=3, columnspacing=0.3)
        make_interactive()
        return fig

    def plot_marker_data(self, title: str, y_label: str, kine_var: str, fig_num: int, add_sd: bool = True,
                         clip_graph: bool = False, marker: str = '') -> matplotlib.figure.Figure:
        """Plot raw and smoothed kinematic (position, velocity, acceleration) marker data, allowing the addition of
        confidence bounds."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_raw = marker_graph_init(ax, getattr(self.raw.means, kine_var), y_label, x_data=self.frames,
                                      color='indigo', marker=marker)
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames,
                                          color='limegreen')

        if add_sd:
            marker_graph_add_cov(ax, getattr(self.smoothed.means, kine_var), getattr(self.smoothed.covars, kine_var),
                                 self.filtered_frames, 'limegreen')
        if clip_graph:
            for c_ax in ax:
                c_ax.set_xlim(self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        else:
            add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        fig.legend((lines_raw[0], lines_smoothed[0]), ('Raw', 'Smoothed'), 'upper right', labelspacing=0.1, ncol=2,
                   columnspacing=0.3)
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_marker_data_diff(self, title: str, y_label: str, fig_num: int) -> matplotlib.figure.Figure:
        """Plot smoothed marker positions - raw marker positions."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_smoothed = marker_graph_init(ax, self.smoothed_pos_diff, y_label, x_data=self.filtered_frames,
                                           color='limegreen')
        fig.legend([lines_smoothed[0]], ['Smoothed'], 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_marker_data_diff_hist(self, title: str, x_label: str, fig_num: int, clip_graph: bool = True) \
            -> matplotlib.figure.Figure:
        """Plot histogram of smoothed marker positions - raw marker positions."""
        if clip_graph:
            endpts = np.zeros((2,), dtype=np.int)

            if self.vicon_endpts[0] > self.filtered.endpts[0]:
                endpts[0] = self.vicon_endpts[0] - self.filtered.endpts[0]

            if self.vicon_endpts[1] < self.filtered.endpts[1]:
                endpts[1] = self.vicon_endpts[1] - self.filtered.endpts[0]
            else:
                endpts[1] = self.filtered.endpts[1] - self.filtered.endpts[0]

            smoothed_diff = self.smoothed_pos_diff[endpts[0]:endpts[1]]
        else:
            smoothed_diff = self.smoothed_pos_diff

        fig = plt.figure(num=fig_num)
        ax = fig.subplots(1, 3, sharey=True)
        lines_smoothed = marker_diff_his_init(ax, smoothed_diff, x_label, 'limegreen')
        fig.legend([lines_smoothed[0]], ['Smoothed'], 'upper right', labelspacing=0.1)
        marker_graph_title(fig, title)
        make_interactive()
        return fig
