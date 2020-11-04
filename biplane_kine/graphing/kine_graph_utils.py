"""This package contains utilities for creating plots of segment kinematics."""

import matplotlib.lines
import matplotlib.axes
import matplotlib.ticker as ticker
import numpy as np
from typing import List, Collection, Tuple
from pythonGraphingLibrary import plotUtils


def kine_graph_init(ax: matplotlib.axes.Axes, marker_data: np.ndarray, y_label: str, x_data: np.ndarray,
                    plot_args=None) -> List[matplotlib.lines.Line2D]:
    """Plot each column of marker_data ((n,3) numpy array) onto the axes provided in ax.

    Additionally, visually format each Axes and include a y_label on 2nd Axes. **kwargs passed to matplotlib plot().
    """
    if plot_args is None:
        plot_args = [{}] * marker_data.shape[1]

    lines = []
    for dim in range(marker_data.shape[1]):
        current_line, = ax.plot(x_data, marker_data[:, dim], **plot_args[dim])
        lines.append(current_line)

    plotUtils.update_spines(ax)
    plotUtils.update_xticks(ax, font_size=8)
    plotUtils.update_yticks(ax, fontsize=8)
    ax.margins(x=0, y=0.05)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
    plotUtils.update_xlabel(ax, 'Frame Number', font_size=10)
    plotUtils.update_ylabel(ax, y_label, font_size=10)

    return lines


def kine_graph_add(ax: matplotlib.axes.Axes, marker_data: np.ndarray, x_data: np.ndarray, plot_args=None) \
        -> List[matplotlib.lines.Line2D]:
    """Plot each column of marker_data ((n,3) numpy array) onto the axes provided in ax.

    Additionally, visually format each Axes and include a y_label on 2nd Axes. **kwargs passed to matplotlib plot().
    """
    if plot_args is None:
        plot_args = [{}] * marker_data.shape[1]

    lines = []
    for dim in range(marker_data.shape[1]):
        current_line, = ax.plot(x_data, marker_data[:, dim], **plot_args[dim])
        lines.append(current_line)

    return lines


def plot_marker_cluster_avail(ax: matplotlib.axes.Axes, marker_data: np.ndarray, frame_nums: np.ndarray,
                              marker_names: Collection[str], vicon_endpts: np.ndarray, **kwargs) \
        -> Tuple[List[matplotlib.lines.Line2D], List[matplotlib.lines.Line2D]]:
    """Create plot showing the presence of each marker in a marker cluster."""
    marker_ints = np.arange(len(marker_names))
    markers_present = ~np.any(np.isnan(marker_data), 2)
    lines_present = []
    lines_absent = []
    for i in marker_ints:
        marker_present = markers_present[i]
        line_present = ax.plot(frame_nums[marker_present], np.ones(np.count_nonzero(marker_present)) * (i+1), 'gs',
                               ms=6, ls='', **kwargs)
        line_absent = ax.plot(frame_nums[~marker_present], np.ones(np.count_nonzero(~marker_present)) * (i + 1), 'rs',
                              ms=6, ls='', **kwargs)
        lines_present.append(line_present[0])
        lines_absent.append(line_absent[0])

    ax.set_yticks(marker_ints + 1)
    ax.set_yticklabels(marker_names)
    plotUtils.update_spines(ax)
    plotUtils.update_xticks(ax, font_size=8)
    plotUtils.update_yticks(ax, fontsize=8)
    plotUtils.update_xlabel(ax, 'Frame Number', font_size=10)

    ax.axvline(vicon_endpts[0] + 1)
    ax.axvline(vicon_endpts[1])

    return lines_present, lines_absent
