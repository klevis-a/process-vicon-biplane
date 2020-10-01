import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections.abc import Iterable
from matplotlib import axes
from pythonGraphingLibrary import plotUtils


def marker_graph_title(fig, title):
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(top=0.94)


def marker_graph_init(ax, marker_data, y_label, x_data, **kwargs):
    lines = []
    for n in range(3):
        current_line, = ax[n].plot(x_data, marker_data[:, n], **kwargs)
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
    return lines


def marker_graph_add(ax, marker_data, x_data, **kwargs):
    lines = []
    for n in range(3):
        current_line, = ax[n].plot(x_data, marker_data[:, n], **kwargs)
        lines.append(current_line)
    return lines


def marker_graph_add_cov(ax, marker_data, cov_data, x_data, color):
    polys = []
    for n in range(3):
        sd = np.sqrt(cov_data[:, n])
        poly = ax[n].fill_between(x_data, marker_data[:, n] - sd, marker_data[:, n] + sd, color=color, alpha=0.2)
        polys.append(poly)
    return polys


def marker_diff_his_init(ax, filtered_diff, x_label, color):
    polygons_filtered = []
    for n in range(3):
        current_filtered_diff = filtered_diff[:, n]
        _, _, patches_filtered = ax[n].hist(current_filtered_diff[~np.isnan(current_filtered_diff)], bins=20,
                                            histtype='step', color=color)
        polygons_filtered.append(patches_filtered[0])
        plotUtils.update_spines(ax[n])
        plotUtils.update_xticks(ax[n], font_size=8)
        plotUtils.update_yticks(ax[n], fontsize=8)

        if n == 1:
            plotUtils.update_xlabel(ax[n], x_label, font_size=10)
        elif n == 0:
            plotUtils.update_ylabel(ax[n], 'Instances', font_size=10)

    return polygons_filtered


def marker_diff_his_add(axs, smoothed_diff, color):
    polygons_smoothed = []
    for (n, ax) in enumerate(axs):
        current_smoothed_diff = smoothed_diff[:, n]
        _, _, patches_smoothed = ax.hist(current_smoothed_diff[~np.isnan(current_smoothed_diff)], bins=20,
                                         histtype='step', color=color)
        polygons_smoothed.append(patches_smoothed[0])
    return polygons_smoothed


def cov_trend_graph_init(ax, variance_data, x_data, y_labels, process_func, **kwargs):
    lines = []
    # iterate over pos, vel, acc
    for i in range(3):
        # iterate over dimension
        dim_lines = []
        for j in range(3):
            line, = ax[i, j].plot(x_data, process_func(variance_data[i][:, j]), **kwargs)
            dim_lines.append(line)
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
    return lines


def cov_trend_graph_add(ax, variance_data, x_data, process_func, **kwargs):
    lines = []
    # iterate over pos, vel, acc
    for i in range(3):
        # iterate over dimension
        dim_lines = []
        for j in range(3):
            line, = ax[i, j].plot(x_data, process_func(variance_data[i][:, j]), **kwargs)
            dim_lines.append(line)
        lines.append(dim_lines)
    return lines


def add_vicon_start_stop(axs, start_i, end_i):
    if isinstance(axs, Iterable):
        for ax in axs:
            add_vicon_start_stop(ax, start_i, end_i)
    elif isinstance(axs, axes.Axes):
        if start_i <= axs.get_xlim()[0]:
            axs.set_xlim(left=start_i-5)
        if end_i >= axs.get_xlim()[1]:
            axs.set_xlim(right=end_i+5)
        axs.axvline(start_i)
        axs.axvline(end_i)
    else:
        raise TypeError('Only arrays of matplotlib.axes objects can be utilized.')
