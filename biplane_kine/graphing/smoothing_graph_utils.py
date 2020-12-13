"""This package provides utility functions for graphing marker data."""

import matplotlib.figure
import matplotlib.lines
import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections.abc import Iterable
from typing import List, Sequence, Callable, Any
from matplotlib import axes
from biplane_kine.graphing import plot_utils


def marker_graph_title(fig: matplotlib.figure.Figure, title: str) -> None:
    """Add title to figure."""
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(top=0.94)


def marker_graph_init(ax: np.ndarray, marker_data: np.ndarray, y_label: str, x_data: np.ndarray, **kwargs) \
        -> List[matplotlib.lines.Line2D]:
    """Plot each column of marker_data ((n,3) numpy array) onto the axes provided in ax.

    Additionally, visually format each Axes and include a y_label on 2nd Axes. **kwargs passed to matplotlib plot().
    """

    lines = []
    for n in range(3):
        current_line, = ax[n].plot(x_data, marker_data[:, n], **kwargs)
        lines.append(current_line)
        plot_utils.update_spines(ax[n])
        plot_utils.update_xticks(ax[n], font_size=8)
        plot_utils.update_yticks(ax[n], fontsize=8)
        ax[n].margins(x=0, y=0.05)
        ax[n].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

        if n == 2:
            plot_utils.update_xlabel(ax[n], 'Frame Number', font_size=10)
        elif n == 1:
            plot_utils.update_ylabel(ax[n], y_label, font_size=10)
    return lines


def marker_graph_add(ax: np.ndarray, marker_data: np.ndarray, x_data: np.ndarray, **kwargs) \
        -> List[matplotlib.lines.Line2D]:
    """Plot each column of marker_data ((n,3) numpy array) onto the axes provided in ax.

    **kwargs passed to matplotlib plot().
    """
    lines = []
    for n in range(3):
        current_line, = ax[n].plot(x_data, marker_data[:, n], **kwargs)
        lines.append(current_line)
    return lines


def marker_graph_add_cov(ax: np.ndarray, marker_data: np.ndarray, cov_data: np.ndarray, x_data: np.ndarray,
                         color: str) -> List[matplotlib.collections.PolyCollection]:
    """Plot a 1 standard deviation shaded confidence bound for each column of marker_data ((n,3) numpy array) as
    specified by the variance in cov_data ((n,3) numpy array) onto the axes provided in ax."""
    polys = []
    for n in range(3):
        sd = np.sqrt(cov_data[:, n])
        poly = ax[n].fill_between(x_data, marker_data[:, n] - sd, marker_data[:, n] + sd, color=color, alpha=0.2)
        polys.append(poly)
    return polys


def marker_diff_his_init(ax: np.ndarray, filtered_diff: np.ndarray, x_label: str, color: str) \
        -> List[matplotlib.patches.Polygon]:
    """Histogram plot each column of filtered_diff ((n,3) numpy array) onto the axes provided in ax.

    Additionally, visually format each Axes and include a x_label on 2nd Axes.
    """

    polygons_filtered = []
    for n in range(3):
        current_filtered_diff = filtered_diff[:, n]
        _, _, patches_filtered = ax[n].hist(current_filtered_diff[~np.isnan(current_filtered_diff)], bins=20,
                                            histtype='step', color=color)
        polygons_filtered.append(patches_filtered[0])
        plot_utils.update_spines(ax[n])
        plot_utils.update_xticks(ax[n], font_size=8)
        plot_utils.update_yticks(ax[n], fontsize=8)

        if n == 1:
            plot_utils.update_xlabel(ax[n], x_label, font_size=10)
        elif n == 0:
            plot_utils.update_ylabel(ax[n], 'Instances', font_size=10)

    return polygons_filtered


def marker_diff_his_add(axs: np.ndarray, smoothed_diff: np.ndarray, color: str) -> List[matplotlib.patches.Polygon]:
    """Histogram plot each column of filtered_diff ((n,3) numpy array) onto the axes provided in axs."""
    polygons_smoothed = []
    for (n, ax) in enumerate(axs):
        current_smoothed_diff = smoothed_diff[:, n]
        _, _, patches_smoothed = ax.hist(current_smoothed_diff[~np.isnan(current_smoothed_diff)], bins=20,
                                         histtype='step', color=color)
        polygons_smoothed.append(patches_smoothed[0])
    return polygons_smoothed


def cov_trend_graph_init(ax: np.ndarray, variance_data: Any, x_data: np.ndarray, y_labels: Sequence[str],
                         process_func: Callable, **kwargs) -> List[List[matplotlib.lines.Line2D]]:
    """Plot each kinematic variable/dimension combination contained in variance_data (tuple of 3 kinematic variables,
    each comprised of a (n, 3) numpy array) onto the 3 rows (kinematic variable) and 3 columns (dimension) contained in
    ax.

    Apply process_func to each kinematic variable/dimension before plotting. Additionally, visually format each axes
    and include a y_label for the first column of each row. **kwargs passed to matplotlib plot().
    """

    lines = []
    # iterate over kinematic variable
    for i in range(3):
        # iterate over dimension
        dim_lines = []
        for j in range(3):
            line, = ax[i, j].plot(x_data, process_func(variance_data[i][:, j]), **kwargs)
            dim_lines.append(line)
            plot_utils.update_spines(ax[i, j])
            plot_utils.update_xticks(ax[i, j], font_size=8)
            plot_utils.update_yticks(ax[i, j], fontsize=8)
            ax[i, j].margins(x=0, y=0.05)
            ax[i, j].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

            if i == 2 and j == 1:
                plot_utils.update_xlabel(ax[i, j], 'Frame Number', font_size=10)

            if j == 0:
                plot_utils.update_ylabel(ax[i, j], y_labels[i], font_size=10)
        lines.append(dim_lines)
    return lines


def cov_trend_graph_add(ax: np.ndarray, variance_data: Any, x_data: np.ndarray, process_func: Callable, **kwargs) \
        -> List[List[matplotlib.lines.Line2D]]:
    """Plot each kinematic variable/dimension combination contained in variance_data (tuple of 3 kinematic variables,
    each comprised of a (n, 3) numpy array) onto the 3 rows (kinematic variable) and 3 columns (dimension) contained in
    ax.

    Apply process_func to each kinematic variable/dimension before plotting. **kwargs passed to matplotlib plot().
    """

    lines = []
    # iterate over kinematic variable
    for i in range(3):
        # iterate over dimension
        dim_lines = []
        for j in range(3):
            line, = ax[i, j].plot(x_data, process_func(variance_data[i][:, j]), **kwargs)
            dim_lines.append(line)
        lines.append(dim_lines)
    return lines


def add_vicon_start_stop(axs: np.ndarray, start_i: int, end_i: int) -> None:
    """Add a vertical line to each axes contained in axs (could be multi-dimensional) at start_i, and end_i.

    Expand the the axes limits if start_i or end_i does not lie within the Axes.get_xlim().
    """

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
