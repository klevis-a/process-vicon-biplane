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


def marker_graph_add_cov(ax, marker_data, cov_data, x_data, color):
    polys = []
    for n in range(3):
        sd = np.sqrt(cov_data[x_data, n])
        poly = ax[n].fill_between(x_data + 1, marker_data[x_data, n] - sd, marker_data[x_data, n] + sd, color=color,
                                  alpha=0.2)
        polys.append(poly)
    return polys


def make_interactive(multiple=True):
    mplcursors.cursor(multiple=multiple)


def marker_diff_his(filtered_diff, smoothed_diff, title, x_label, fig_num, colors):
    fig = plt.figure(num=fig_num)
    ax = fig.subplots(1, 3, sharey=True)
    polygons_filtered = []
    polygons_smoothed = []
    for n in range(3):
        current_filtered_diff = filtered_diff[:, n]
        current_smoothed_diff = smoothed_diff[:, n]
        _, _, patches_filtered = ax[n].hist(current_filtered_diff[~np.isnan(current_filtered_diff)], bins=20,
                                            histtype='step', color=colors[0])
        polygons_filtered.append(patches_filtered[0])
        _, _, patches_smoothed = ax[n].hist(current_smoothed_diff[~np.isnan(current_smoothed_diff)], bins=20,
                                            histtype='step', color=colors[1])
        polygons_smoothed.append(patches_smoothed[0])
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
    return fig, ax, polygons_filtered, polygons_smoothed


def sd_trend_graph(var_filtered, var_smoothed, title, y_labels, fig_num, endpts, styles=('r-', 'g-')):
    fig = plt.figure(num=fig_num)
    ax = fig.subplots(3, 3, sharex='all', sharey='row')
    lines = []
    x_data = np.arange(var_filtered[0].shape[0])
    # iterate over pos, vel, acc
    for i in range(3):
        # iterate over dimension
        dim_lines = []
        for j in range(3):
            filtered_line, = ax[i, j].plot(x_data + 1, np.sqrt(var_filtered[i][:, j]), styles[0])
            smoothed_line, = ax[i, j].plot(x_data + 1, np.sqrt(var_smoothed[i][:, j]), styles[1])
            dim_lines.append((filtered_line, smoothed_line))
            ax[i, j].axvline(endpts[0] + 1)
            ax[i, j].axvline(endpts[1] + 1)
            plotUtils.update_spines(ax[i, j])
            plotUtils.update_xticks(ax[i, j], font_size=8)
            plotUtils.update_yticks(ax[i, j], fontsize=8)
            ax[i, j].margins(x=0, y=0.05)
            ax[i, j].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

            if i == 2 and j == 1:
                plotUtils.update_xlabel(ax[i, j], 'Frame Number', font_size=10)

            if j == 0:
                plotUtils.update_ylabel(ax[i, j], y_labels[i], font_size=10)
        lines.append(dim_lines)

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    return fig, ax, lines


def plot_marker_data(trial_name, marker_name, y_label, md_raw, md_filtered, md_smoothed, var, var_smoothed, kine_var,
                     fig_num, endpts):
    x_data = np.arange(md_raw.pos.shape[0]) if endpts is None else np.arange(endpts[0], endpts[1])
    fig, ax, lines_raw = marker_graph_init(getattr(md_raw, kine_var), trial_name + ' ' + marker_name, y_label,
                                           fig_num=fig_num, x_data=x_data)
    lines_filtered = marker_graph_add(ax, getattr(md_filtered, kine_var), x_data, 'r-')
    lines_smoothed = marker_graph_add(ax, getattr(md_smoothed, kine_var), x_data, 'g-')
    marker_graph_add_cov(ax, getattr(md_filtered, kine_var), getattr(var, kine_var), x_data, 'red')
    marker_graph_add_cov(ax, getattr(md_smoothed, kine_var), getattr(var_smoothed, kine_var), x_data, 'green')
    fig.legend((lines_raw[0], lines_filtered[0], lines_smoothed[0]), ('Raw', 'Filtered', 'Smoothed'), 'upper right',
               labelspacing=0.1)
    make_interactive()


def plot_marker_data_diff(trial_name, marker_name, y_label, filtered_diff, smoothed_diff, fig_num, endpts):
    x_data = np.arange(filtered_diff.shape[0]) if endpts is None else np.arange(endpts[0], endpts[1])
    fig, ax, lines_filtered = marker_graph_init(filtered_diff, trial_name + ' ' + marker_name, y_label, fig_num=fig_num,
                                                x_data=x_data, style='r-')
    lines_smoothed = marker_graph_add(ax, smoothed_diff, x_data, 'g-')
    fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
    make_interactive()


def plot_marker_data_diff_hist(trial_name, marker_name, x_label, filtered_diff, smoothed_diff, fig_num, endpts):
    if endpts is None:
        fig, _, lines_filtered, lines_smoothed = marker_diff_his(filtered_diff, smoothed_diff,
                                                                 trial_name + ' ' + marker_name, x_label, fig_num,
                                                                 ['red', 'green'])
    else:
        fig, _, lines_filtered, lines_smoothed = marker_diff_his(filtered_diff[endpts[0]:endpts[1]],
                                                                 smoothed_diff[endpts[0]:endpts[1]],
                                                                 trial_name + ' ' + marker_name, x_label, fig_num,
                                                                 ['red', 'green'])
    fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
    make_interactive()


def plot_sd(trial_name, marker_name, y_labels, variances_filtered, variances_smooth, fig_num, endpts):
    sd_trend_graph(variances_filtered, variances_smooth, trial_name + ' ' + marker_name + ' SD', y_labels, fig_num,
                   endpts)


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
        self.covs = None
        self.covs_smooth = None
        self.variances = None
        self.variances_smooth = None

        if do_filter:
            self.filter()

        if do_plot:
            self.plot()

    @classmethod
    def extract_variances(cls, covs):
        # here we iterate over each dimension (x, y, z)
        variances = []
        for i in range(3):
            variance = np.vstack([cov[:, i, i] for cov in covs]).T
            variances.append(variance)
        return variances

    def filter(self):
        kf = LinearKalmanFilter1D(dt=self.dt, discrete_white_noise_var=10000, r=1, p=np.diag([0.5, 0.5, 0.5]))
        self.filtered_md, self.smoothed_md, self.covs, self.covs_smooth = kf.filter_trial_marker(self.raw_md.pos)
        self.variances = LinearKalmanFilter1D.FilterOutput(*MarkerPlotter.extract_variances(self.covs))
        self.variances_smooth = LinearKalmanFilter1D.FilterOutput(*MarkerPlotter.extract_variances(self.covs_smooth))

    def plot(self, all_frames=True, biplane_frames=True, plot_diff=True):
        y_labels = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s$^2$)']
        attrs = ['pos', 'vel', 'acc']
        num_figs = len(y_labels)

        filtered_diff = None
        smoothed_diff = None
        if plot_diff:
            filtered_diff = self.filtered_md.pos - self.raw_md.pos
            smoothed_diff = self.smoothed_md.pos - self.raw_md.pos

        current_fig_num = 0
        if all_frames:
            for (fig_num, (y_label, attr)) in enumerate(zip(y_labels, attrs)):
                plot_marker_data(self.trial_name, self.marker_name, y_label, self.raw_md, self.filtered_md,
                                 self.smoothed_md, self.variances, self.variances_smooth, attr,
                                 current_fig_num + fig_num, None)
            if plot_diff:
                plot_marker_data_diff(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                      smoothed_diff, current_fig_num + num_figs, None)
                plot_marker_data_diff_hist(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                           smoothed_diff, current_fig_num + num_figs + 1, None)

        current_fig_num = (current_fig_num + num_figs + (2 if plot_diff is True else 0)) if all_frames is True else 0
        if biplane_frames:
            for (fig_num, (y_label, attr)) in enumerate(zip(y_labels, attrs)):
                plot_marker_data(self.trial_name, self.marker_name, y_label, self.raw_md, self.filtered_md,
                                 self.smoothed_md, self.variances, self.variances_smooth, attr,
                                 current_fig_num + fig_num, self.trial.vicon_endpts)
            if plot_diff:
                plot_marker_data_diff(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                      smoothed_diff, current_fig_num + num_figs, self.trial.vicon_endpts)
                plot_marker_data_diff_hist(self.trial_name, self.marker_name, 'Filtering Effect (mm)', filtered_diff,
                                           smoothed_diff, current_fig_num + num_figs + 1, self.trial.vicon_endpts)

        current_fig_num = (current_fig_num + num_figs +
                           (2 if plot_diff is True else 0)) if biplane_frames is True else 0
        y_labels = ['Pos (mm)', 'Vel (mm/s)', 'Acc (mm/s$^2$)']
        plot_sd(self.trial_name, self.marker_name, y_labels, self.variances, self.variances_smooth, current_fig_num,
                self.trial.vicon_endpts)
