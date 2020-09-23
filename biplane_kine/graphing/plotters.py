import numpy as np
from biplane_kine.graphing.graph_utils \
    import (marker_graph_init, marker_graph_add, add_vicon_start_stop, make_interactive, marker_diff_his_init,
            marker_diff_his_add, marker_graph_add_cov, cov_trend_graph_init, cov_trend_graph_add, MplStyle)


class LabeledMarkerPloter:
    def __init__(self, trial, marker_name):
        self.trial = trial
        self.marker_name = marker_name
        self.marker_pos_labeled = self.trial.marker_data_labeled(marker_name)
        self.frame_nums = np.arange(self.marker_pos_labeled.shape[0]) + 1

    def plot(self):
        fig, ax, _ = marker_graph_init(self.marker_pos_labeled, self.trial.trial_name + ' ' + self.marker_name,
                                       'Pos (mm)', fig_num=0, x_data=self.frame_nums)
        add_vicon_start_stop(ax, self.trial.vicon_endpts[0] + 1, self.trial.vicon_endpts[1])
        make_interactive()
        return fig


class LabeledFilledMarkerPlotter(LabeledMarkerPloter):
    def __init__(self, trial, marker_name):
        super().__init__(trial, marker_name)
        self.marker_pos_filled = self.trial.marker_data_filled(marker_name)

    def plot(self):
        fig, ax, _ = marker_graph_init(self.marker_pos_filled, self.trial.trial_name + ' ' + self.marker_name,
                                       'Pos (mm)', fig_num=0, x_data=self.frame_nums, style=MplStyle('red'))
        marker_graph_add(ax, self.marker_pos_labeled, self.frame_nums, MplStyle('indigo', marker='.'))
        add_vicon_start_stop(ax, self.trial.vicon_endpts[0] + 1, self.trial.vicon_endpts[1])
        make_interactive()
        return [fig]


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
        y_labels_corr = ['Pos Vel Corr', 'Pos Acc Corr', 'Vel Acc Corr']
        attrs = ['pos', 'vel', 'acc']
        title = self.trial_name + ' ' + self.marker_name

        current_fig_num = 0
        figs = []
        pos_fig = self.plot_marker_data(title, y_labels[0], 'pos', current_fig_num, add_sd=False, clip_graph=True)
        current_fig_num += 1
        figs.append(pos_fig)

        for (y_label, attr) in zip(y_labels, attrs):
            kine_fig = self.plot_marker_data(title, y_label, attr, current_fig_num)
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

    def plot_marker_data(self, title, y_label, kine_var, fig_num, add_sd=True, clip_graph=False, marker=''):
        fig, ax, lines_raw = marker_graph_init(getattr(self.raw.means, kine_var), title, y_label, fig_num,
                                               x_data=self.frames, style=MplStyle('indigo', marker=marker))
        lines_filtered = marker_graph_add(ax, getattr(self.filtered.means, kine_var), self.filtered_frames,
                                          style=MplStyle('red'))
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames,
                                          style=MplStyle('limegreen'))

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
        make_interactive()
        return fig

    def plot_marker_data_smooth(self, title, y_label, kine_var, fig_num):
        fig, ax, lines_smoothed = marker_graph_init(getattr(self.smoothed.means, kine_var), title, y_label, fig_num,
                                                    x_data=self.filtered_frames, style=MplStyle('limegreen'))
        fig.legend([lines_smoothed[0]], ['Smoothed'], 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig

    def plot_marker_data_diff(self, title, y_label, fig_num):
        fig, ax, lines_filtered = marker_graph_init(self.filtered_pos_diff, title, y_label, fig_num,
                                                    x_data=self.filtered_frames, style=MplStyle('red'))
        lines_smoothed = marker_graph_add(ax, self.smoothed_pos_diff, self.filtered_frames, style=MplStyle('limegreen'))
        fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig

    def plot_marker_data_diff_hist(self, title, x_label, fig_num, clip_graph=True):
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

        fig, ax, lines_filtered = marker_diff_his_init(filtered_diff, title, x_label, fig_num, 'red')
        lines_smoothed = marker_diff_his_add(ax, smoothed_diff, 'limegreen')

        fig.legend((lines_filtered[0], lines_smoothed[0]), ('Filtered', 'Smoothed'), 'upper right', labelspacing=0.1)
        make_interactive()
        return fig

    def plot_cov(self, title, y_labels, fig_num):
        fig, ax, lines_filtered = cov_trend_graph_init(self.filtered.covars, self.filtered_frames, title, y_labels,
                                                       fig_num, np.sqrt, style=MplStyle('red'))
        lines_smooth = cov_trend_graph_add(ax, self.smoothed.covars, self.filtered_frames, np.sqrt,
                                           style=MplStyle('limegreen'))
        fig.legend((lines_filtered[0][0], lines_smooth[0][0]), ('Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=2)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig

    def plot_corr(self, title, y_labels, fig_num):
        fig, ax, lines_filtered = cov_trend_graph_init(self.filtered.corrs, self.filtered_frames, title, y_labels,
                                                       fig_num, lambda x: x, style=MplStyle('red'))
        lines_smooth = cov_trend_graph_add(ax, self.smoothed.corrs, self.filtered_frames, lambda x: x,
                                           style=MplStyle('limegreen'))
        fig.legend((lines_filtered[0][0], lines_smooth[0][0]), ('Filtered', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=2)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig


class SmoothingOutputPlotter(SmoothingDebugPlotter):
    def __init__(self, trial_name, marker_name, raw, filled, filtered, smoothed, vicon_endpts):
        super().__init__(trial_name, marker_name, raw, filtered, smoothed, vicon_endpts)
        self.filled = filled

    def plot(self):
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

    def plot_pos_data(self, title, y_label, kine_var, fig_num, add_sd=True, clip_graph=False):
        fig, ax, lines_filled = marker_graph_init(self.filled.means.pos, title, y_label, fig_num, x_data=self.frames,
                                                  style=MplStyle('red'))
        lines_raw = marker_graph_add(ax, self.raw.means.pos, self.frames, style=MplStyle('indigo', marker='.'))
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames,
                                          style=MplStyle('limegreen', lw=1))

        if add_sd:
            marker_graph_add_cov(ax, self.smoothed.means.pos, self.smoothed.covars.pos, self.filtered_frames,
                                 'limegreen')
        if clip_graph:
            for c_ax in ax:
                c_ax.set_xlim(self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        else:
            add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])

        fig.suptitle(title, x=0.25, fontsize=11, fontweight='bold')
        fig.legend((lines_raw[0], lines_filled[0], lines_smoothed[0]), ('Raw', 'Filled', 'Smoothed'), 'upper right',
                   labelspacing=0.1, ncol=3, columnspacing=0.3)
        make_interactive()
        return fig

    def plot_marker_data(self, title, y_label, kine_var, fig_num, add_sd=True, clip_graph=False, marker=''):
        fig, ax, lines_raw = marker_graph_init(getattr(self.raw.means, kine_var), title, y_label, fig_num,
                                               x_data=self.frames, style=MplStyle('indigo', marker=marker))
        lines_smoothed = marker_graph_add(ax, getattr(self.smoothed.means, kine_var), self.filtered_frames,
                                          MplStyle('limegreen'))

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
        make_interactive()
        return fig

    def plot_marker_data_diff(self, title, y_label, fig_num):
        fig, ax, lines_smoothed = marker_graph_init(self.smoothed_pos_diff, title, y_label, fig_num,
                                                    x_data=self.filtered_frames, style=MplStyle('limegreen'))
        fig.legend([lines_smoothed[0]], ['Smoothed'], 'upper right', labelspacing=0.1)
        add_vicon_start_stop(ax, self.vicon_frame_endpts[0], self.vicon_frame_endpts[1])
        make_interactive()
        return fig

    def plot_marker_data_diff_hist(self, title, x_label, fig_num, clip_graph=True):
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

        fig, ax, lines_smoothed = marker_diff_his_init(smoothed_diff, title, x_label, fig_num, 'limegreen')

        fig.legend([lines_smoothed[0]], ['Smoothed'], 'upper right', labelspacing=0.1)
        make_interactive()
        return fig
