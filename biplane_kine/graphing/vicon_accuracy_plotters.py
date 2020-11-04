"""A module that provides plotters for Vicon markers that have been tracked via biplane fluoroscopy to ascertain the
spatiotemporal syncing accuracy between the Vicon and biplane fluoroscopy systems ."""

import numpy as np
from typing import Sequence, List
import matplotlib.figure
import matplotlib.pyplot as plt
from biplane_kine.graphing.common_graph_utils import make_interactive
from biplane_kine.graphing.smoothing_graph_utils import marker_graph_init, marker_graph_add, marker_graph_title
from biplane_kine.graphing.vicon_accuracy_graph_utils import marker_diff_graph
from biplane_kine.vicon_biplane_diff import BiplaneViconDiff, BiplaneViconSmoothDiff


class ViconAccuracyPlotter:
    """Plotter that enables comparing marker position data as measured by Vicon (raw/labeled) and biplane
    fluoroscopy.

    Attributes
    ----------
    trial_name: str
        Trial identifier.
    marker_name: str
        Marker identifier.
    biplane_vicon_diff: biplane_kine.vicon_biplane_diff.BiplaneViconDiff
        Contains differences between marker position data as measured via Vicon and biplane fluoroscopy, and associated
        statistics.
    vicon_data_raw: numpy.ndarray, (N, 3)
        Raw (labeled) Vicon marker data positions.
    vicon_indices: numpy.ndarray, (N, )
        Zero-based indices for the Vicon frames.
    vicon_frames: numpy.ndarray, (N, )
        One-based indices for the Vicon frames - useful for plotting.
    biplane_data: numpy.ndarray, (N, )
        Marker data positions as measured via biplane fluoroscopy.
    """

    def __init__(self, trial_name: str, marker_name: str, biplane_vicon_diff: BiplaneViconDiff):
        self.trial_name = trial_name
        self.marker_name = marker_name
        self.biplane_vicon_diff = biplane_vicon_diff
        self.vicon_data_raw = biplane_vicon_diff.vmd_fluoro
        self.vicon_indices = np.arange(self.vicon_data_raw.shape[0])
        self.vicon_frames = self.vicon_indices + 1
        self.biplane_data = np.full((self.vicon_data_raw.shape[0], 3), np.nan)
        self.biplane_data[biplane_vicon_diff.biplane_marker_data.indices, :] = \
            biplane_vicon_diff.biplane_marker_data.data

        # biplane_vicon_diff.raw_diff contains differences between Vicon and biplane only at the frames where the
        # marker was tracked via biplane fluoroscopy. This doesn't include the entire biplane fluoroscopy trial. This
        # is fine, but when graphing it's useful to see the entire trial timeline. Matplotlib doesn't graph NaN values
        # so we first create a matrix with all NaN values that encompasses the entire timeline of the biplane
        # fluoroscopy trial. Then copy the Vicon/biplane fluoroscopy differences to the frames where the marker was
        # measured via biplane fluoroscopy.
        self._diff_raw = np.full((self.vicon_data_raw.shape[0], 3), np.nan)
        self._diff_raw[biplane_vicon_diff.biplane_marker_data.indices, :] = biplane_vicon_diff.raw_diff
        self._diff_raw_scalar = np.full((self.vicon_data_raw.shape[0],), np.nan)
        self._diff_raw_scalar[biplane_vicon_diff.biplane_marker_data.indices] = biplane_vicon_diff.raw_diff_scalar

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot figures quantifying differences in marker position as measured via Vicon versus biplane fluoroscopy.

        Figure 0: Trend plots of marker position tracked via biplane fluoroscopy vs Vicon (raw/labeled) overlayed
        Figure 1: Trend plots of the difference between marker position data as tracked via biplane fluoroscopy vs
                  Vicon (raw/labeled)
        """
        title = self.trial_name + ' ' + self.marker_name
        figs = []

        # plot biplane and vicon marker data together
        acc_vicon_fig = self.plot_biplane_vicon(title, 0, 'vicon_data_raw', 'Raw')
        figs.append(acc_vicon_fig)

        # plot difference
        diff_fig = self.plot_diff(title, 1, ['_diff_raw', '_diff_raw_scalar'], 'raw')
        figs.append(diff_fig)

        return figs

    def plot_biplane_vicon(self, title: str, fig_num: int, vicon_field: str, vicon_type: str) \
            -> matplotlib.figure.Figure:
        """Plot overlayed marker position data as measured via Vicon and biplane fluoroscopy."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(3, 1, sharex=True)
        lines_vicon = marker_graph_init(ax, getattr(self, vicon_field), 'Distance (mm)', self.vicon_frames,
                                        color='limegreen', marker='.', lw=1, ms=2)
        lines_biplane = marker_graph_add(ax, self.biplane_data, self.vicon_frames, color='indigo', marker='.')

        fig.legend((lines_biplane[0], lines_vicon[0]), ('Biplane', vicon_type + ' Vicon'), 'upper right', ncol=3,
                   columnspacing=0.3, handlelength=1.0)
        marker_graph_title(fig, title)
        make_interactive()
        return fig

    def plot_diff(self, title: str, fig_num: int, vicon_fields: Sequence[str], diff_field: str) \
            -> matplotlib.figure.Figure:
        """Plot difference between marker position data as measured via Vicon and biplane fluoroscopy."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        lines_xyz = marker_diff_graph(ax[0], getattr(self, vicon_fields[0]), 'Distance (mm)', x_label=None,
                                      x_data=self.vicon_frames)
        line_scalar = marker_diff_graph(ax[1], getattr(self, vicon_fields[1]), 'Distance (mm)', None,
                                        self.vicon_frames, color='indigo')
        fig.legend(lines_xyz + line_scalar, ('X', 'Y', 'Z', '| |'), loc='lower center', ncol=4, columnspacing=0.3,
                   handlelength=1.0)
        fig.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout(pad=1.0, h_pad=0, rect=(0, 0.015, 1, 1))

        # add RMS, MAE, Max for each individual x, y, z
        text_align = [(0.01, 0.99, 'left', 'top'), (0.99, 0.99, 'right', 'top'), (0.01, 0.01, 'left', 'bottom')]
        cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for n in range(3):
            xyz_text = 'RMS: {:.2f} MAE: {:.2f} Max: {:.2f}'\
                .format(getattr(self.biplane_vicon_diff, diff_field + '_rms')[n],
                        getattr(self.biplane_vicon_diff, diff_field + '_mae')[n],
                        getattr(self.biplane_vicon_diff, diff_field + '_max')[n])
            ax[0].text(text_align[n][0], text_align[n][1], xyz_text, ha=text_align[n][2], va=text_align[n][3],
                       transform=fig.transFigure, fontweight='bold',
                       bbox=dict(ec=cc[n], fc='None', boxstyle='round', lw=2))

        # add RMS, MAE, Max for scalar
        scalar_text = 'RMS: {:.2f} MAE: {:.2f} Max: {:.2f}'\
            .format(getattr(self.biplane_vicon_diff, diff_field + '_rms_scalar'),
                    getattr(self.biplane_vicon_diff, diff_field + '_mae_scalar'),
                    getattr(self.biplane_vicon_diff, diff_field + '_max_scalar'))
        ax[0].text(0.99, 0.01, scalar_text, ha='right', va='bottom', transform=fig.transFigure, fontweight='bold',
                   bbox=dict(ec='indigo', fc='None', boxstyle='round', lw=2))
        make_interactive()
        return fig


class ViconAccuracySmoothingPlotter(ViconAccuracyPlotter):
    """Plotter that enables comparing marker position data as measured by Vicon (then smoothed) and biplane
    fluoroscopy.

    Attributes
    ----------
    vicon_data_smoothed: numpy.ndarray, (N, 3)
        Smoothed Vicon marker data position.
    """

    def __init__(self, trial_name: str, marker_name: str, biplane_vicon_smooth_diff: BiplaneViconSmoothDiff):
        super().__init__(trial_name, marker_name, biplane_vicon_smooth_diff)
        self.vicon_data_smoothed = biplane_vicon_smooth_diff.smoothed_vmd_fluoro

        # see comment in base class as to why the operation below is performed
        self._diff_smoothed = np.full((self.vicon_data_raw.shape[0], 3), np.nan)
        self._diff_smoothed[biplane_vicon_smooth_diff.biplane_marker_data.indices, :] = \
            biplane_vicon_smooth_diff.smoothed_diff
        self._diff_smoothed_scalar = np.full((self.vicon_data_raw.shape[0],), np.nan)
        self._diff_smoothed_scalar[biplane_vicon_smooth_diff.biplane_marker_data.indices] = \
            biplane_vicon_smooth_diff.smoothed_diff_scalar

    def plot(self) -> List[matplotlib.figure.Figure]:
        """Plot figures quantifying differences in marker position as measured via Vicon, Vicon (then smoothed), and
        biplane fluoroscopy.

        Figure 0: Trend plots of marker position tracked via biplane fluoroscopy vs Vicon (raw/labeled) overlayed
        Figure 1: Trend plots of the difference between marker position data as tracked via biplane fluoroscopy vs
                  Vicon (raw/labeled)
        Figure 2: Trend plots of marker position tracked via biplane fluoroscopy vs Vicon (smoothed) overlayed
        Figure 3: Trend plots of the difference between marker position data as tracked via biplane fluoroscopy vs
                  Vicon (smoothed)
        Figure 4: Figure 1 and 3 overlayed on top of each other
        """
        title = self.trial_name + ' ' + self.marker_name
        figs = super().plot()

        # plot biplane and vicon marker data together
        acc_vicon_fig = self.plot_biplane_vicon(title, 2, 'vicon_data_smoothed', 'Smooth ')
        figs.append(acc_vicon_fig)

        # plot difference
        diff_fig = self.plot_diff(title, 3, ['_diff_smoothed', '_diff_smoothed_scalar'], 'smoothed')
        figs.append(diff_fig)

        # plot all differences in the same plot
        diff_all_fig = self.plot_all_diff(title, 4)
        figs.append(diff_all_fig)

        return figs

    def plot_all_diff(self, title: str, fig_num: int) -> matplotlib.figure.Figure:
        """Overlay plot differences in marker position as measured via Vicon and biplane fluoroscopy, and differences
        in marker position as measured Vicon (then smoothed) and biplane fluoroscopy."""
        fig = plt.figure(num=fig_num)
        ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        lines_xyz_raw = marker_diff_graph(ax[0], self._diff_raw, 'Distance (mm)', x_label=None,
                                          x_data=self.vicon_frames, ls='--')
        line_scalar_raw = marker_diff_graph(ax[1], self._diff_raw_scalar, 'Distance (mm)', 'Frame Number',
                                            self.vicon_frames, color='indigo', ls=':')
        # reset colors
        ax[0].set_prop_cycle(None)
        lines_xyz_smooth = marker_diff_graph(ax[0], self._diff_smoothed, 'Distance (mm)', x_label=None,
                                             x_data=self.vicon_frames)
        line_scalar_smooth = marker_diff_graph(ax[1], self._diff_smoothed_scalar, 'Distance (mm)', 'Frame Number',
                                               self.vicon_frames, color='indigo')
        leg1 = fig.legend(lines_xyz_raw + line_scalar_raw, ('X (Raw)', 'Y', 'Z', '$\\mid \\mid$'), loc='lower left',
                          handletextpad=0.1, ncol=4, columnspacing=0.5, handlelength=1.0, bbox_to_anchor=(0.0, 0.0))
        fig.legend(lines_xyz_smooth + line_scalar_smooth, ('X (Smooth)', 'Y', 'Z', '$\\mid \\mid$'), loc='lower right',
                   handletextpad=0.1, ncol=4, columnspacing=0.5, handlelength=1.0, bbox_to_anchor=(1.0, 0.0))
        fig.add_artist(leg1)
        fig.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout(pad=1.0, h_pad=0, rect=(0, 0.015, 1, 1))

        # add RMS, MAE, Max
        raw_text = 'RMS: {:.2f} MAE: {:.2f} Max: {:.2f}'\
            .format(self.biplane_vicon_diff.raw_rms_scalar, self.biplane_vicon_diff.raw_mae_scalar,
                    self.biplane_vicon_diff.raw_max_scalar)
        smooth_text = 'RMS: {:.2f} MAE: {:.2f} Max: {:.2f}'\
            .format(self.biplane_vicon_diff.smoothed_rms_scalar, self.biplane_vicon_diff.smoothed_mae_scalar,
                    self.biplane_vicon_diff.smoothed_max_scalar)
        ax[0].text(0.01, 0.99, raw_text, ha='left', va='top', transform=fig.transFigure, fontweight='bold',
                   bbox=dict(ec='indigo', fc='None', boxstyle='round', ls=':', lw=2))
        ax[0].text(0.99, 0.99, smooth_text, ha='right', va='top', transform=fig.transFigure, fontweight='bold',
                   bbox=dict(ec='indigo', fc='None', boxstyle='round', lw=2))
        make_interactive()
        return fig
