"""A module that provides graphing utilies for Vicon markers that have been tracked via biplane fluoroscopy to ascertain
the spatiotemporal syncing accuracy between the Vicon and biplane fluoroscopy systems ."""

import numpy as np
from typing import Union
import matplotlib.axes
import matplotlib.lines
from pythonGraphingLibrary import plotUtils


def marker_diff_graph(ax: matplotlib.axes.Axes, marker_data: np.ndarray, y_label: str, x_label: Union[str, None],
                      x_data: np.ndarray, **kwargs) -> matplotlib.lines.Line2D:
    """Plot marker_data onto the axes provided in ax.

    Additionally, visually format the axes and include a y_label. **kwargs passed to matplotlib plot().
    """
    lines = ax.plot(x_data, marker_data, **kwargs)
    plotUtils.update_spines(ax)
    plotUtils.update_xticks(ax, font_size=8)
    plotUtils.update_yticks(ax, fontsize=8)
    if x_label:
        plotUtils.update_xlabel(ax, x_label, font_size=10)
    plotUtils.update_ylabel(ax, y_label, font_size=10)
    return lines
