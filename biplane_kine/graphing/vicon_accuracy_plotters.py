import numpy as np
import matplotlib.pyplot as plt
from biplane_kine.graphing.common_graph_utils import MplStyle, make_interactive
from biplane_kine.graphing.smoothing_graph_utils import marker_graph_init, marker_graph_add, marker_graph_title
from biplane_kine.graphing.vicon_accuracy_graph_utils import marker_diff_graph


class ViconAccuracySmoothingPlotter:
    def __init__(self, trial_name, marker_name, acc_marker_data, vicon_data):
        self.trial_name = trial_name
        self.marker_name = marker_name
        self.vicon_data = vicon_data
        self.vicon_indices = np.arange(vicon_data.shape[0])
        self.vicon_frames = self.vicon_indices + 1
        self.acc_data = acc_marker_data.data
        self.acc_frames = acc_marker_data.frames
        self.acc_indices = acc_marker_data.indices
        self.diff_data = self.vicon_data[self.acc_indices] - self.acc_data

    def plot(self):
        title = self.trial_name + ' ' + self.marker_name
        figs = []

        # plot biplane and vicon marker data together
        acc_vicon_fig = self.plot_acc_vicon(title, 0)
        figs.append(acc_vicon_fig)

        # plot difference
        diff_fig = self.plot_diff(title, 1)
        figs.append(diff_fig)

        return figs

    def plot_acc_vicon(self, title, fig_num):
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_acc = marker_graph_init(ax, self.acc_data, 'Distance (mm)', x_data=self.acc_frames,
                                      style=MplStyle('indigo', marker='.'))
        lines_vicon = marker_graph_add(ax, self.vicon_data, self.vicon_frames, MplStyle('limegreen', marker='.', lw=1))
        fig.legend((lines_acc[0], lines_vicon[0]), ('Biplane', 'Vicon'), 'upper right', labelspacing=0.1, ncol=3,
                   columnspacing=0.3)
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_diff(self, title, fig_num):
        fig = plt.figure(num=fig_num)
        ax = fig.subplots()
        lines = marker_diff_graph(ax, self.diff_data, 'Distance (mm)', x_data=self.acc_frames)
        fig.legend(lines, ('X', 'Y', 'Z'), 'upper right', labelspacing=0.1, ncol=3, columnspacing=0.3)
        fig.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
        make_interactive()
        return fig
