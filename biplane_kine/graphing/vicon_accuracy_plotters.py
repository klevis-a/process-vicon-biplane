import numpy as np
import matplotlib.pyplot as plt
from biplane_kine.kinematics.vec_ops import extended_dot
from biplane_kine.graphing.common_graph_utils import MplStyle, make_interactive
from biplane_kine.graphing.smoothing_graph_utils import marker_graph_init, marker_graph_add, marker_graph_title
from biplane_kine.graphing.vicon_accuracy_graph_utils import marker_diff_graph


class ViconAccuracyPlotter:
    def __init__(self, trial_name, marker_name, biplane_marker_data, vicon_data):
        self.trial_name = trial_name
        self.marker_name = marker_name
        self.vicon_data_raw = vicon_data
        self.vicon_indices = np.arange(vicon_data.shape[0])
        self.vicon_frames = self.vicon_indices + 1
        self.biplane_data = biplane_marker_data.data
        self.biplane_frames = biplane_marker_data.frames
        self.biplane_indices = biplane_marker_data.indices
        self.diff_raw = self.vicon_data_raw[self.biplane_indices] - self.biplane_data
        self.diff_raw_scalar = np.sqrt(extended_dot(self.diff_raw, self.diff_raw))

    def plot(self):
        title = self.trial_name + ' ' + self.marker_name
        figs = []

        # plot biplane and vicon marker data together
        acc_vicon_fig = self.plot_biplane_vicon(title, 0, 'vicon_data_raw')
        figs.append(acc_vicon_fig)

        # plot difference
        diff_fig = self.plot_diff(title, 1, ['diff_raw', 'diff_raw_scalar'])
        figs.append(diff_fig)

        return figs

    def plot_biplane_vicon(self, title, fig_num, vicon_field):
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_acc = marker_graph_init(ax, self.biplane_data, 'Distance (mm)', x_data=self.biplane_frames,
                                      style=MplStyle('indigo', marker='.'))
        lines_vicon = marker_graph_add(ax, getattr(self, vicon_field), self.vicon_frames,
                                       MplStyle('limegreen', marker='.', lw=1))
        fig.legend((lines_acc[0], lines_vicon[0]), ('Biplane', 'Vicon'), 'upper right', labelspacing=0.1, ncol=3,
                   columnspacing=0.3)
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_diff(self, title, fig_num, vicon_fields):
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        lines_xyz = marker_diff_graph(ax[0], getattr(self, vicon_fields[0]), 'Distance (mm)', x_label=None,
                                      x_data=self.biplane_frames)
        ax[0].legend(lines_xyz, ('X', 'Y', 'Z'), loc='upper right', labelspacing=0.1, ncol=3, columnspacing=0.3)
        line_scalar = marker_diff_graph(ax[1], getattr(self, vicon_fields[1]), 'Distance (mm)', 'Frame Number',
                                        x_data=self.biplane_frames)
        ax[1].legend(line_scalar, ['Magnitude'], loc='upper right')

        fig.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
        make_interactive()
        return fig


class ViconAccuracySmoothingPlotter(ViconAccuracyPlotter):
    def __init__(self, trial_name, marker_name, biplane_marker_data, vicon_data, vicon_data_smoothed):
        super().__init__(trial_name, marker_name, biplane_marker_data, vicon_data)
        self.vicon_data_smoothed = vicon_data_smoothed
        self.diff_smoothed = self.vicon_data_smoothed[self.biplane_indices] - self.biplane_data
        self.diff_smoothed_scalar = np.sqrt(extended_dot(self.diff_smoothed, self.diff_smoothed))

    def plot(self):
        title = self.trial_name + ' ' + self.marker_name
        figs = super().plot()

        # plot biplane and vicon marker data together
        acc_vicon_fig = self.plot_biplane_vicon(title, 2, 'vicon_data_smoothed')
        figs.append(acc_vicon_fig)

        # plot difference
        diff_fig = self.plot_diff(title, 3, ['diff_smoothed', 'diff_smoothed_scalar'])
        figs.append(diff_fig)

        return figs
