import numpy as np
from graphing.graph_utils import marker_graph_init, marker_graph_add, add_vicon_start_stop, make_interactive, \
    marker_diff_his_init, marker_diff_his_add, marker_graph_add_cov, cov_trend_graph_init, cov_trend_graph_add


class LabeledMarkerPloter:
    def __init__(self, trial, marker_name, dt):
        self.trial = trial
        self.marker_name = marker_name
        self.dt = dt
        self.marker_pos_labeled = self.trial.marker_data_labeled(marker_name)
        self.frame_nums = np.arange(self.marker_pos_labeled.shape[0])

    def plot(self):
        fig, ax, _ = marker_graph_init(self.marker_pos_labeled, self.trial.trial_name + ' ' + self.marker_name,
                                       'Pos (mm)', fig_num=0, x_data=self.frame_nums + 1)
        add_vicon_start_stop(ax, self.trial.vicon_endpts[0] + 1, self.trial.vicon_endpts[1])
        make_interactive()
        return fig


class LabeledFilledMarkerPlotter(LabeledMarkerPloter):
    def __init__(self, trial, marker_name, dt):
        super().__init__(trial, marker_name, dt)
        self.marker_pos_filled = self.trial.marker_data_filled(marker_name)

    def plot(self):
        fig, ax, _ = marker_graph_init(self.marker_pos_filled, self.trial.trial_name + ' ' + self.marker_name,
                                       'Pos (mm)', fig_num=0, x_data=self.frame_nums + 1, style='r-')
        marker_graph_add(ax, self.marker_pos_labeled, self.frame_nums + 1, 'b-')
        add_vicon_start_stop(ax, self.trial.vicon_endpts[0] + 1, self.trial.vicon_endpts[1])
        make_interactive()
        return fig


class SmoothingDebugPlotter:
    def __init__(self, trial_name, marker_name, raw, filtered, smoothed, vicon_endpts):
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

    def plot(self, plot_diff=True):
        y_labels = ['Position (mm)', 'Velocity (mm/s)', 'Acceleration (mm/s$^2$)']
        attrs = ['pos', 'vel', 'acc']
        num_figs = len(y_labels)
        title = self.trial_name + ' ' + self.marker_name

        figs = []
        pos_fig = self.plot_marker_data(title, y_labels[0], 'pos', fig_num=0, add_sd=False, clip_graph=True)
        figs.append(pos_fig)

        current_fig_num = 1
        for (fig_num, (y_label, attr)) in enumerate(zip(y_labels, attrs)):
            kine_fig = self.plot_marker_data(title, y_label, attr, current_fig_num + fig_num)
            figs.append(kine_fig)

        if plot_diff:
            trend_diff = self.plot_marker_data_diff(title, 'Filtering Effect (mm)', fig_num=current_fig_num + num_figs)
            hist_diff = self.plot_marker_data_diff_hist(title, 'Filtering Effect (mm)', current_fig_num + num_figs + 1)
            figs.append(trend_diff)
            figs.append(hist_diff)

        current_fig_num = current_fig_num + num_figs + (2 if plot_diff is True else 0)
        var_plot = self.plot_cov(title, y_labels, np.sqrt, current_fig_num)
        y_labels_corr = ['Pos Vel Corr', 'Pos Acc Corr', 'Vel Acc Corr']
        corr_plot = self.plot_cov(title, y_labels_corr, lambda x: x, current_fig_num + 1)
        figs.append(var_plot)
        figs.append(corr_plot)

        return figs

    def plot_marker_data(self, title, y_label, kine_var, fig_num, add_sd=True, clip_graph=False):
        fig, ax, lines_raw = marker_graph_init(getattr(self.raw.means, kine_var), title, y_label, fig_num,
                                               x_data=self.frames)
        lines_filtered = marker_graph_add(ax, getattr(self.filtered.means, kine_var), self.filtered_frames, 'r-')
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames, 'g-')

        if add_sd:
            marker_graph_add_cov(ax, getattr(self.filtered.means, kine_var), getattr(self.filtered.covars, kine_var),
                                 self.filtered_frames, 'red')
            marker_graph_add_cov(ax, getattr(self.smoothed.means, kine_var), getattr(self.smoothed.covars, kine_var),
                                 self.filtered_frames, 'green')
        if clip_graph:
            for c_ax in ax:
                c_ax.set_xlim(self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        else:
            add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        fig.legend((lines_raw[0], lines_filtered[0], lines_smoothed[0]), ('Raw', 'Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1)
        make_interactive()
        return fig

    def plot_marker_data_diff(self, title, y_label, fig_num):
        fig, ax, lines_filtered = marker_graph_init(self.filtered_pos_diff, title, y_label, fig_num,
                                                    x_data=self.filtered_frames, style='r-')
        lines_smoothed = marker_graph_add(ax, self.smoothed_pos_diff, self.filtered_frames, 'g-')
        fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig

    def plot_marker_data_diff_hist(self, title, x_label, fig_num, clip_graph=True):
        if clip_graph:
            filtered_diff = self.filtered_pos_diff[self.vicon_endpts[0]:self.vicon_endpts[1]]
            smoothed_diff = self.smoothed_pos_diff[self.vicon_endpts[0]:self.vicon_endpts[1]]
        else:
            filtered_diff = self.filtered_pos_diff
            smoothed_diff = self.smoothed_pos_diff

        fig, ax, lines_filtered = marker_diff_his_init(filtered_diff, title, x_label, fig_num, 'red')
        lines_smoothed = marker_diff_his_add(ax, smoothed_diff, 'green')

        fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
        make_interactive()
        return fig

    def plot_cov(self, title, y_labels, process_func, fig_num):
        fig, ax, lines_filtered = cov_trend_graph_init(self.filtered.covars, self.filtered_frames, title, y_labels,
                                                       fig_num, process_func, 'r-')
        lines_smooth = cov_trend_graph_add(ax, self.smoothed.covars, self.filtered_frames, process_func, 'g-')
        fig.legend((lines_filtered[0][0], lines_smooth[0][0]), ('Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=2)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig
