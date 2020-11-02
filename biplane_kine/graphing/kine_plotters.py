import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import List, Dict
from .common_graph_utils import make_interactive
from .smoothing_graph_utils import marker_graph_init, marker_graph_add
from .kine_graph_utils import kine_graph_init, kine_graph_add, plot_marker_cluster_avail


class RawSmoothedKineTorsoPlotter:
    """Plotter for torso kinematics derived from labeled, filled, and smoothed skin marker data.

    Attributes
    ----------
    torso_pos_labeled: numpy.ndarray (N, 3)
        Torso position trajectory derived from labeled (raw) skin marker data
    torso_eul_labeled: numpy.ndarray (N, 3)
        Torso orientation trajectory (expressed as an Euler angle sequence) derived from labeled (raw) skin marker data
    torso_pos_filled: numpy.ndarray (N, 3)
        Torso position trajectory derived from filled (raw) skin marker data
    torso_eul_filled: numpy.ndarray (N, 3)
        Torso orientation trajectory (expressed as an Euler angle sequence) derived from filled (raw) skin marker data
    torso_pos_smoothed: numpy.ndarray (N, 3)
        Torso position trajectory derived from smoothed (raw) skin marker data
    torso_eul_smoothed: numpy.ndarray (N, 3)
        Torso orientation trajectory (expressed as an Euler angle sequence) derived from smoothed (raw) skin marker data
    frame_nums: numpy.ndarray (N,)
        Vicon frame numbers associated with torso trajectory
    trial_name: str
        Name of trial being plotted
    """
    def __init__(self, trial_name: str, torso_pos_labeled: np.ndarray, torso_eul_labeled: np.ndarray,
                 torso_pos_filled: np.ndarray, torso_eul_filled: np.ndarray, torso_pos_smoothed: np.ndarray,
                 torso_eul_smoothed: np.ndarray, frame_nums: np.ndarray):
        self.torso_pos_labeled = torso_pos_labeled
        self.torso_eul_labeled = torso_eul_labeled
        self.torso_pos_filled = torso_pos_filled
        self.torso_eul_filled = torso_eul_filled
        self.torso_pos_smoothed = torso_pos_smoothed
        self.torso_eul_smoothed = torso_eul_smoothed
        self.frame_nums = frame_nums
        self.trial_name = trial_name

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot torso position and orientation trajectory as derived from labeled (raw), filled, and smoothed skin
        marker data.

        Figures:
        1. Torso position broken out into 3 separate subplots for each spatial dimension
        2. Torso orientation broken out into 3 separate subplots for each spatial dimension
        3. Torso position plotted on one axes with different colors for each spatial dimension
        4. Torso orientation plotted on one axes with different colors for each spatial dimension
        """
        figs = []

        # Figure 1, position in 3 subplots
        pos_fig_sub = self.plot_subplots(0, self.trial_name + ' Torso Intrinsic Position', 'Position (mm)',
                                         self.torso_pos_labeled, self.torso_pos_filled, self.torso_pos_smoothed)
        figs.append(pos_fig_sub)

        # Figure 2, orientation in 3 subplots
        eul_fig_sub = self.plot_subplots(1, self.trial_name + ' Torso Intrinsic Euler Angles', 'Angle (deg)',
                                         self.torso_eul_labeled, self.torso_eul_filled, self.torso_eul_smoothed)
        figs.append(eul_fig_sub)

        # Figure 3, position in one axes
        pos_fig_one = self.plot_one_axes(2, self.trial_name + ' Torso Intrinsic Position', 'Position (mm)',
                                         self.torso_pos_labeled, self.torso_pos_filled, self.torso_pos_smoothed,
                                         {'labeled': 'Labeled (X)', 'filled': 'Filled (Y)', 'smoothed': 'Smoothed (Z)'})
        figs.append(pos_fig_one)

        # Figure 3, position in one axes
        eul_fig_one = self.plot_one_axes(3, self.trial_name + ' Torso Intrinsic Euler Angles', 'Angle (deg)',
                                         self.torso_eul_labeled, self.torso_eul_filled, self.torso_eul_smoothed,
                                         {'labeled': 'Labeled (Flex/Ext)', 'filled': 'Filled (Lat Flex)',
                                          'smoothed': 'Smoothed (Axial)'})
        figs.append(eul_fig_one)

        return figs

    def plot_subplots(self, fig_num: int, title: str, y_label: str, labeled: np.ndarray, filled: np.ndarray,
                      smoothed: np.ndarray) -> matplotlib.figure.Figure:
        """Plot torso position or orientation into 3 separate subplots for each spatial dimension."""
        fig = plt.figure(fig_num)
        axs = fig.subplots(3, 1, sharex=True)
        labeled_lines = marker_graph_init(axs, labeled, y_label, self.frame_nums, color='blue')
        filled_lines = marker_graph_add(axs, filled, self.frame_nums, color='red')
        smoothed_lines = marker_graph_add(axs, smoothed, self.frame_nums, color='green')
        plt.tight_layout()
        fig.suptitle(title)
        fig.legend((labeled_lines[0], filled_lines[0], smoothed_lines[0]), ('Labeled', 'Filled', 'Smoothed'),
                   ncol=3, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='upper left')
        make_interactive()
        return fig

    def plot_one_axes(self, fig_num: int, title: str, y_label: str, labeled: np.ndarray, filled: np.ndarray,
                      smoothed: np.ndarray, legend_entries: Dict[str, str]) -> matplotlib.figure.Figure:
        """Plot torso position or orientation on one axes with different colors for each spatial dimension."""
        fig = plt.figure(fig_num)
        ax = fig.subplots(1, 1)
        labeled_lines = kine_graph_init(ax, labeled, y_label, self.frame_nums, [{'ls': '', 'marker': 'o', 'ms': 2,
                                                                                 'fillstyle': 'none', 'mew': 0.5}] * 3)
        ax.set_prop_cycle(None)
        filled_lines = kine_graph_add(ax, filled, self.frame_nums, [{'ls': '-', 'lw': 0.75}] * 3)
        ax.set_prop_cycle(None)
        smoothed_lines = kine_graph_add(ax, smoothed, self.frame_nums, [{'ls': '-'}] * 3)
        plt.tight_layout()
        fig.suptitle(title, x=0.7)
        fig.legend((labeled_lines[0], smoothed_lines[2], filled_lines[1]),
                   (legend_entries['labeled'], legend_entries['smoothed'], legend_entries['filled']),
                   ncol=2, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='upper left')
        make_interactive()
        return fig


class MarkerClusterAvailPlotter:
    """Plotter for visually determining when markers in a marker clustser are present in a trial.

    Attributes
    ----------
    marker_data: numpy.ndarray (M, N, 3)
        marker position data for all M markers and N frames in a trial
    marker_names: List of str
        list of marker names in marker cluster
    vicon_endpts: array_like (2,)
        The frame indices (endpoints) of the Vicon trial that correspond to the endpoints of the reciprocal
        biplane fluoroscopy trial.
    """

    def __init__(self, marker_data: np.ndarray, marker_names: List[str], vicon_endpts: np.ndarray, trial_name: str):
        self.marker_data = marker_data
        self.marker_names = marker_names
        self.vicon_endpts = vicon_endpts
        self.trial_name = trial_name
        self.frame_nums = np.arange(self.marker_data.shape[1]) + 1

    def plot(self) -> List[matplotlib.figure.Figure]:
        fig = plt.figure()
        ax = fig.subplots(1, 1)
        present_lines, absent_lines = plot_marker_cluster_avail(ax, self.marker_data, self.frame_nums,
                                                                self.marker_names, self.vicon_endpts)
        plt.tight_layout()
        fig.suptitle(self.trial_name)
        plt.subplots_adjust(top=0.95)
        fig.legend((present_lines[0], absent_lines[0]), ('Present', 'Absent'), ncol=2, handlelength=0.75,
                   handletextpad=0.5, columnspacing=1.0, loc='upper right')
        make_interactive()
        return [fig]
