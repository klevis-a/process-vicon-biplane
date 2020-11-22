import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import List, Sequence, Dict, Tuple, Union
from pythonGraphingLibrary import plotUtils
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

    def __init__(self, marker_data: np.ndarray, marker_names: Sequence[str], vicon_endpts: np.ndarray, trial_name: str):
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


class MarkerClusterFillPlotter(MarkerClusterAvailPlotter):
    """Plotter for visually demonstrating filled gaps and the markers utilized to fill them.

    Attributes
    ----------
    gaps_filled: dict of marker names to gaps
        Dictionary containing the gaps that were filled for each marker
    source_markers: dict of marker names to source markers
        Dictionary containing the source markers that were utilized to fill gaps for each marker
    filled_data: dict of marker names to filled marker trajectories
        Dictionary containing the filled marker trajectory for each marker
    sfs_data: dict of marker names to smoothed/filled/smoothed marker trajectories
        Dictionary containing the smoothed/filled/smoothed marker trajectory for each marker
    """
    def __init__(self, trial_name: str, marker_data: np.ndarray, marker_names: Sequence[str],
                 gaps_filled: Dict[str, Sequence[Tuple[int, int]]], source_markers: Dict[str, Sequence[str]],
                 filled_data: Dict[str, np.ndarray], sfs_data: Dict[str, np.ndarray], vicon_endpts: np.ndarray):
        super().__init__(marker_data, marker_names, vicon_endpts, trial_name)
        self.gaps_filled = gaps_filled
        self.source_markers = source_markers
        self.filled_data = filled_data
        self.sfs_data = sfs_data

    def plot(self) -> List[matplotlib.figure.Figure]:
        fig = super().plot()[0]
        ax_fig = fig.axes[0]

        # add gap demarcation lines and source marker names for marker that was filled
        for (idx, marker_name) in enumerate(self.marker_names):
            if marker_name in self.gaps_filled:
                gaps = self.gaps_filled[marker_name]
                for gap in gaps:
                    ax_fig.vlines([gap[0] + 1, gap[1]], ymin=(idx + 1) - 0.2, ymax=(idx + 1) + 0.2, linewidths=6,
                                  colors=(1, 0.5, 0))
                max_gap_idx = np.argmax([gap[1] - gap[0] for gap in gaps])
                ax_fig.text((gaps[max_gap_idx][0] + gaps[max_gap_idx][1])/2, idx + 1 + 0.25,
                            ','.join(self.source_markers[marker_name]), horizontalalignment='center',
                            verticalalignment='bottom', fontweight='bold')

        # create a figure for each marker that was filled
        figs = self.plot_filled_trajectory(2)

        return [fig] + figs

    def plot_filled_trajectory(self, fig_start) -> List[matplotlib.figure.Figure]:
        figs = []
        for (idx, (marker_name, filled_marker_data)) in enumerate(self.filled_data.items()):
            fig = plt.figure(fig_start + idx)
            ax = fig.subplots(3, 1)
            filled_lines = marker_graph_init(ax, filled_marker_data, 'Position (mm)', self.frame_nums, color='red')
            smoothed_lines = marker_graph_add(ax, self.marker_data[self.marker_names.index(marker_name)],
                                              self.frame_nums, color='blue')
            sfs_lines = marker_graph_add(ax, self.sfs_data[marker_name], self.frame_nums, color='green')
            highlight_sfs = np.full_like(self.sfs_data[marker_name], np.nan)
            gaps = self.gaps_filled[marker_name]
            for gap in gaps:
                highlight_sfs[gap[0]:gap[1], :] = self.sfs_data[marker_name][gap[0]:gap[1], :]
            high_sfs_lines = marker_graph_add(ax, highlight_sfs, self.frame_nums, color='orange')
            for sub_ax in ax:
                sub_ax.set_xlim(left=1)
            plt.tight_layout()
            fig.suptitle(self.trial_name + ' ' + marker_name, x=0.75)
            fig.legend((filled_lines[0], smoothed_lines[0], sfs_lines[0], high_sfs_lines[0]),
                       ('Filled Raw', 'Smoothed', 'SFS', 'Filled Smoothed'), ncol=4, handlelength=0.75,
                       handletextpad=0.25, columnspacing=0.5, loc='lower left')
            make_interactive()
            figs.append(fig)
        return figs


class TorsoTrajComparisonPlotter:
    """Torso trajectory plotter comparing previously filled, smoothed, smoothed/filled, and smoothed/filled/smoothed
    torso kinematics.

    Attributes
    ----------
    trial_name: str
        Name of trial
    prev_filled: tuple of numpy.ndarray (N, 3)
        Torso position and Euler angles derived from marker position data that had been filled in Vicon
    smoothed: tuple of numpy.ndarray (N, 3)
        Torso position and Euler angles derived from smoothed marker position data
    filled: tuple of numpy.ndarray (N, 3)
        Torso position and Euler angles dervied from smoothed then filled marker position data
    sfs: tuple of numpy.ndarray (N, 3)
        Torso position and Euler angles dervied from smoothed, filled, then smoothed again (lightly)
        marker position data
    frame_nums: numpy.ndarray (N,)
        Frame numbers for the trial
    vicon_endpts: array_like (2,)
        The frame indices (endpoints) of the Vicon trial that correspond to the endpoints of the reciprocal
        biplane fluoroscopy trial.
    """
    def __init__(self, trial_name, prev_filled, smoothed, filled, sfs, vicon_endpts):
        self.trial_name = trial_name
        self.prev_filled = prev_filled
        self.smoothed = smoothed
        self.filled = filled
        self.sfs = sfs
        self.frame_nums = np.arange(self.sfs[0].shape[0]) + 1
        self.vicon_endpts = vicon_endpts

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot torso position and orientation trajectory as derived from previously filled, smoothed,
        smoothed then filled, and smoothed/filled/smoothed skin marker position.

        Figures:
        1. Torso position broken out into 3 separate subplots for each spatial dimension
        2. Torso orientation broken out into 3 separate subplots for each spatial dimension
        """
        figs = []
        # Figure 1: Position
        fig = self.plot_kine_var(1, self.trial_name, ('X (mm)', 'Y (mm)', 'Z (mm)'), self.prev_filled[0],
                                 self.smoothed[0], self.filled[0], self.sfs[0])
        figs.append(fig)

        # Figure 2: Orientation
        fig = self.plot_kine_var(2, self.trial_name, ('Flex/Ext (deg)', 'Lat Flex (deg)', 'Axial (deg)'),
                                 self.prev_filled[1], self.smoothed[1], self.filled[1], self.sfs[1])
        figs.append(fig)

        return figs

    def plot_kine_var(self, fig_num: int, title: str, y_labels: Sequence[str], prev_filled: np.ndarray,
                      smoothed: np.ndarray, filled: np.ndarray, sfs: np.ndarray) -> matplotlib.figure.Figure:
        """Plot torso position or orientation on one axes with different colors for each spatial dimension."""
        fig = plt.figure(fig_num)
        ax = fig.subplots(3, 1)
        prev_filled_lines = marker_graph_init(ax, prev_filled, '', self.frame_nums, color='red')
        smoothed_lines = marker_graph_add(ax, smoothed, self.frame_nums, color='blue')
        smoothed_filled_lines = marker_graph_add(ax, filled, self.frame_nums, ls=':', lw=2, color='green')
        sfs_lines = marker_graph_add(ax, sfs, self.frame_nums, color='green')
        for idx, sub_ax in enumerate(ax):
            plotUtils.update_ylabel(sub_ax, y_labels[idx], font_size=10)
            sub_ax.axvline(self.vicon_endpts[0])
            sub_ax.axvline(self.vicon_endpts[1])
            sub_ax.set_xlim(left=1)
        plt.tight_layout()
        fig.suptitle(title)
        fig.legend((prev_filled_lines[0], smoothed_lines[0], smoothed_filled_lines[0], sfs_lines[0]),
                   ('Prev Filled', 'Smoothed', 'Smoothed/Filled', 'SFS'),
                   ncol=4, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left')
        make_interactive()
        return fig


class RawSmoothSegmentPlotter:
    """Plotter for torso kinematics derived from labeled, filled, and smoothed skin marker data.

    Attributes
    ----------
    pos_raw: numpy.ndarray (N, 3)
        Raw position trajectory
    eul_raw: numpy.ndarray (N, 3)
        Raw orientation trajectory (expressed as an Euler angle sequence)
    pos_smooth: numpy.ndarray (N, 3)
        Smoothed position trajectory
    eul_smooth: numpy.ndarray (N, 3)
        Smoothed orientation trajectory (expressed as an Euler angle sequence)
    vel: numpy.ndarray (N, 3)
        Smoothed linear velocity
    ang_vel: numpy.ndarray (N, 3)
        Smoothed angular velocity
    frame_nums: numpy.ndarray (N,)
        Biplane fluoroscopy frame numbers
    trial_name: str
        Name of trial being plotted
    segment_name: str
        Name of segment being plotted
    euler_legend: List of str
        Legend specifying the sequence of Euler rotation names
    fig_num_start: int
        Starting figure number
    """
    def __init__(self, trial_name: str, segment_name: str, pos_raw: np.ndarray, eul_raw: np.ndarray,
                 pos_smooth: np.ndarray, eul_smooth: np.ndarray, vel: np.ndarray, ang_vel: np.ndarray,
                 frame_nums: np.ndarray, euler_legend: Sequence[str], pos_legend: Union[Sequence[str], None] = None,
                 fig_num_start: int = 0):
        self.pos_raw = pos_raw
        self.eul_raw = eul_raw
        self.pos_smooth = pos_smooth
        self.eul_smooth = eul_smooth
        self.vel = vel
        self.ang_vel = ang_vel
        self.frame_nums = frame_nums
        self.trial_name = trial_name
        self.segment_name = segment_name
        self.euler_legend = euler_legend
        self.pos_legend = ['X', 'Y', 'Z'] if pos_legend is None else pos_legend
        self.fig_num_start = fig_num_start

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot raw and smoothed position and orientation.

        Figures:
        1. Position broken out into 3 separate subplots for each spatial dimension
        2. Orientation broken out into 3 separate subplots for each spatial dimension
        3. Velocity broken out into 3 separate subplots for each spatial dimension
        4. Angular velocity broken out into 3 separate subplots for each spatial dimension
        5. Position plotted on one axes with different colors for each spatial dimension
        6. Orientation plotted on one axes with different colors for each spatial dimension
        7. Velocity plotted on one axes with different colors for each spatial dimension
        8. Angular velocity plotted on one axes with different colors for each spatial dimension
        """
        figs = []

        title_prefix = self.trial_name + ' ' + self.segment_name + ' '
        # Figure 1, position in 3 subplots
        pos_fig_sub = self.plot_subplots(self.fig_num_start,  title_prefix + 'Position (mm)', self.pos_raw,
                                         self.pos_smooth, self.pos_legend)
        figs.append(pos_fig_sub)

        # Figure 2, orientation in 3 subplots
        eul_fig_sub = self.plot_subplots(self.fig_num_start + 1, title_prefix + 'Euler Angles (deg)', self.eul_raw,
                                         self.eul_smooth, self.euler_legend)
        figs.append(eul_fig_sub)

        # Figure 3, velocity in 3 subplots
        vel_fig_sub = self.plot_subplots_vel(self.fig_num_start + 2, title_prefix + 'Velocity', 'Velocity (mm/s)',
                                             self.vel)
        figs.append(vel_fig_sub)

        # Figure 4, angular velocity in 3 subplots
        ang_vel_fig_sub = self.plot_subplots_vel(self.fig_num_start + 3, title_prefix + 'Angular Velocity',
                                                 'Angular Velocity (deg/s)', self.ang_vel)
        figs.append(ang_vel_fig_sub)

        # Figure 5, position in one axes
        pos_fig_one = self.plot_one_axes(self.fig_num_start + 4, title_prefix + 'Position', 'Position (mm)',
                                         self.pos_raw, self.pos_smooth, self.pos_legend)
        figs.append(pos_fig_one)

        # Figure 6, orientation in one axes
        eul_fig_one = self.plot_one_axes(self.fig_num_start + 5, title_prefix + 'Euler Angles', 'Angle (deg)',
                                         self.eul_raw, self.eul_smooth, self.euler_legend)
        figs.append(eul_fig_one)

        # Figure 7, velocity in one axes
        vel_fig_one = self.plot_one_axes_vel(self.fig_num_start + 6, title_prefix + 'Velocity', 'Velocity (mm/s)',
                                             self.vel, self.pos_legend)
        figs.append(vel_fig_one)

        # Figure 8, angular velocity in one axes
        ang_vel_fig_one = self.plot_one_axes_vel(self.fig_num_start + 7, title_prefix + 'Angular Velocity',
                                                 'Angular Velocity (deg/s)', self.ang_vel, self.pos_legend)
        figs.append(ang_vel_fig_one)

        return figs

    def plot_subplots(self, fig_num: int, title: str, raw: np.ndarray, smoothed: np.ndarray,
                      axes_lbl_entries: Sequence[str]) -> matplotlib.figure.Figure:
        """Plot position or orientation into 3 separate subplots for each spatial dimension."""
        fig = plt.figure(fig_num)
        axs = fig.subplots(3, 1, sharex=True)
        raw_lines = marker_graph_init(axs, raw, '', self.frame_nums, color='red')
        for idx, ax in enumerate(axs):
            plotUtils.update_ylabel(ax, axes_lbl_entries[idx], font_size=10)
        smoothed_lines = marker_graph_add(axs, smoothed, self.frame_nums, color='green')
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        fig.suptitle(title)
        fig.legend((raw_lines[0], smoothed_lines[0]), ('Raw', 'Smoothed'), ncol=2, handlelength=0.75,
                   handletextpad=0.25, columnspacing=0.5, loc='lower left')
        make_interactive()
        return fig

    def plot_one_axes(self, fig_num: int, title: str, y_label: str, raw: np.ndarray, smoothed: np.ndarray,
                      legend_entries: Sequence[str]) -> matplotlib.figure.Figure:
        """Plot position or orientation on one axes with different colors for each spatial dimension."""
        fig = plt.figure(fig_num)
        ax = fig.subplots(1, 1)
        raw_lines = kine_graph_init(ax, raw, y_label, self.frame_nums, [{'ls': ':', 'lw': 2}] * 3)
        ax.set_prop_cycle(None)
        smoothed_lines = kine_graph_add(ax, smoothed, self.frame_nums, [{'ls': '-'}] * 3)
        plt.tight_layout()
        fig.suptitle(title, x=0.7)
        legend_text = ('Raw (' + legend_entries[0] + ')', 'Smoothed (' + legend_entries[1] + ')',
                       'Smoothed (' + legend_entries[2] + ')')
        fig.legend((raw_lines[0], smoothed_lines[1], smoothed_lines[2]), legend_text, ncol=3, handlelength=0.75,
                   handletextpad=0.25, columnspacing=0.5, loc='lower left')
        make_interactive()
        return fig

    def plot_subplots_vel(self, fig_num: int, title: str, y_label: str, vel: np.ndarray) -> matplotlib.figure.Figure:
        """Plot velocity into 3 separate subplots for each spatial dimension."""
        fig = plt.figure(fig_num)
        axs = fig.subplots(3, 1, sharex=True)
        marker_graph_init(axs, vel, y_label, self.frame_nums, color='blue')
        plt.tight_layout()
        fig.suptitle(title)
        make_interactive()
        return fig

    def plot_one_axes_vel(self, fig_num: int, title: str, y_label: str, vel: np.ndarray,
                          legend_entries: Sequence[str]) -> matplotlib.figure.Figure:
        """Plot velocity on one axes with different colors for each spatial dimension."""
        fig = plt.figure(fig_num)
        ax = fig.subplots(1, 1)
        kine_graph_init(ax, vel, y_label, self.frame_nums, [{}] * 3)
        plt.tight_layout()
        fig.suptitle(title, x=0.7)
        fig.legend(legend_entries, ncol=3, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left')
        make_interactive()
        return fig
