import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplcursors
import numpy as np
from matplotlib import rcParams
from pythonGraphingLibrary import plotUtils

from smoothing.kalman_filtering import LinearKalmanFilter1D


def init_graphing():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    matplotlib.use('Qt5Agg')


def marker_graph_init(marker_data, title, y_label, fig_num, x_data, style='b-'):
    fig = plt.figure(num=fig_num)
    ax = fig.subplots(3, 1, sharex=True)
    lines = []
    for n in range(3):
        current_line, = ax[n].plot(x_data + 1, marker_data[x_data, n], style, markersize=2)
        lines.append(current_line)
        plotUtils.update_spines(ax[n])
        plotUtils.update_xticks(ax[n], font_size=8)
        plotUtils.update_yticks(ax[n], fontsize=8)
        ax[n].margins(x=0, y=0.05)
        ax[n].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

        if n == 2:
            plotUtils.update_xlabel(ax[n], 'Frame Number', font_size=10)
        elif n == 1:
            plotUtils.update_ylabel(ax[n], y_label, font_size=10)

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    return fig, ax, lines


def marker_graph_add(ax, marker_data, x_data, style):
    lines = []
    for n in range(3):
        current_line, = ax[n].plot(x_data + 1, marker_data[x_data, n], style)
        lines.append(current_line)
    return lines


def make_interactive(multiple=True):
    mplcursors.cursor(multiple=multiple)


def marker_diff_his(filtered_diff, smoothed_diff, title, x_label, fig_num, colors):
    fig = plt.figure(num=fig_num)
    ax = fig.subplots(1, 3, sharey=True)
    for n in range(3):
        current_filtered_diff = filtered_diff[:, n]
        current_smoothed_diff = smoothed_diff[:, n]
        _, _, patches_filtered = ax[n].hist(current_filtered_diff[~np.isnan(current_filtered_diff)], bins=20,
                                            histtype='step', color=colors[0])
        _, _, patches_smoothed = ax[n].hist(current_smoothed_diff[~np.isnan(current_smoothed_diff)], bins=20,
                                            histtype='step', color=colors[1])
        plotUtils.update_spines(ax[n])
        plotUtils.update_xticks(ax[n], font_size=8)
        plotUtils.update_yticks(ax[n], fontsize=8)

        if n == 1:
            plotUtils.update_xlabel(ax[n], x_label, font_size=10)
        elif n == 0:
            plotUtils.update_ylabel(ax[n], 'Instances', font_size=10)

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    return fig, ax, patches_filtered[0], patches_smoothed[0]


def plot_marker_data(trial_name, marker_name, y_label, md_raw, md_filtered, md_smoothed, kine_var, fig_num, endpts):
    x_data = np.arange(md_raw.pos.shape[0]) if endpts is None else np.arange(endpts[0], endpts[1])
    fig, ax, lines_raw = marker_graph_init(getattr(md_raw, kine_var), trial_name + ' ' + marker_name, y_label,
                                           fig_num=fig_num, x_data=x_data)
    lines_filtered = marker_graph_add(ax, getattr(md_filtered, kine_var), x_data, 'r-')
    lines_smoothed = marker_graph_add(ax, getattr(md_smoothed, kine_var), x_data, 'g-')
    fig.legend((lines_raw[0], lines_filtered[0], lines_smoothed[0]), ('Raw', 'Filtered', 'Smoothed'), 'upper right',
               labelspacing=0.1)
    make_interactive()


def plot_marker_data_diff(trial_name, marker_name, y_label, filtered_diff, smoothed_diff, fig_num, endpts):
    x_data = np.arange(filtered_diff.shape[0]) if endpts is None else np.arange(endpts[0], endpts[1])
    fig, ax, lines_filtered = marker_graph_init(filtered_diff, trial_name + ' ' + marker_name, y_label, fig_num=fig_num,
                                                x_data=x_data, style='r-')
    lines_smoothed = marker_graph_add(ax, smoothed_diff, x_data, 'g-')
    fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)


def marker_data_diff_hist(trial_name, marker_name, x_label, filtered_diff, smoothed_diff, fig_num, endpts):
    if endpts is None:
        fig, _, line_filtered, line_smoothed = marker_diff_his(filtered_diff, smoothed_diff,
                                                               trial_name + ' ' + marker_name, x_label, fig_num,
                                                               ['red', 'green'])
    else:
        fig, _, line_filtered, line_smoothed = marker_diff_his(filtered_diff[endpts[0]:endpts[1]],
                                                               smoothed_diff[endpts[0]:endpts[1]],
                                                               trial_name + ' ' + marker_name, x_label, fig_num,
                                                               ['red', 'green'])
    fig.legend((line_filtered, line_smoothed), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)


class MarkerPlotter:
    def __init__(self, db, trial_name, marker_name, do_filter=True, do_plot=True):
        self.db = db
        self.trial_name = trial_name
        self.marker_name = marker_name
        self.trial = db.loc[trial_name].Trial
        self.dt = self.db.attrs['dt']

        marker_pos = self.trial.marker_data(marker_name)
        marker_vel = np.gradient(marker_pos, self.dt, axis=0)
        marker_acc = np.gradient(marker_vel, self.dt, axis=0)
        self.raw_md = LinearKalmanFilter1D.FilterOutput(marker_pos, marker_vel, marker_acc)
        self.filtered_md = None
        self.smoothed_md = None

        if do_filter:
            self.filter()

        if do_plot:
            self.plot()

    def filter(self):
        kf = LinearKalmanFilter1D(dt=self.dt, discrete_white_noise_var=10000, r=1, p=np.diag([0.5, 0.5, 0.5]))
        self.filtered_md, self.smoothed_md = kf.filter_trial_marker(self.raw_md.pos)

    def plot(self, all_frames=True, biplane_frames=True, plot_diff=True):
        y_labels = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s$^2$)']
        attrs = ['pos', 'vel', 'acc']
        num_figs = len(y_labels)

        if plot_diff:
            filtered_diff = self.filtered_md.pos - self.raw_md.pos
            smoothed_diff = self.smoothed_md.pos - self.raw_md.pos

        current_fig_num = 0
        if all_frames:
            for (fig_num, (y_label, attr)) in enumerate(zip(y_labels, attrs)):
                plot_marker_data(self.trial_name, self.marker_name, y_label, self.raw_md, self.filtered_md,
                                 self.smoothed_md, attr, current_fig_num + fig_num, None)
            if plot_diff:
                plot_marker_data_diff(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                      smoothed_diff, current_fig_num + num_figs, None)
                marker_data_diff_hist(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                      smoothed_diff, current_fig_num + num_figs + 1, None)

        current_fig_num = num_figs + (2 if plot_diff is True else 0) if all_frames is True else 0
        if biplane_frames:
            for (fig_num, (y_label, attr)) in enumerate(zip(y_labels, attrs)):
                plot_marker_data(self.trial_name, self.marker_name, y_label, self.raw_md, self.filtered_md,
                                 self.smoothed_md, attr, current_fig_num + fig_num, self.trial.vicon_endpts)
            if plot_diff:
                plot_marker_data_diff(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                      smoothed_diff, current_fig_num + num_figs, self.trial.vicon_endpts)
                marker_data_diff_hist(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                      smoothed_diff, current_fig_num + num_figs + 1, self.trial.vicon_endpts)
