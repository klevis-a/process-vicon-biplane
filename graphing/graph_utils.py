import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import mplcursors
import numpy as np
from pythonGraphingLibrary import plotUtils


def init_graphing():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    matplotlib.use('Qt5Agg')


def marker_graph_init(marker_data, title, fig_num, x_data=None, style='b-'):
    if x_data is None:
        x_data = np.arange(marker_data.shape[0])
    fig = plt.figure(num=fig_num)
    ax = fig.subplots(3, 1, sharex=True)
    for n in range(3):
        ax[n].plot(x_data + 1, marker_data[x_data, n], style, markersize=2)
        plotUtils.update_spines(ax[n])
        plotUtils.update_xticks(ax[n], font_size=8)
        plotUtils.update_yticks(ax[n], fontsize=8)
        ax[n].margins(x=0, y=0.05)
        ax[n].yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

        if n == 2:
            plotUtils.update_xlabel(ax[n], 'Frame Number', font_size=10)
        elif n == 1:
            plotUtils.update_ylabel(ax[n], 'Position (mm)', font_size=10)

    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.subplots_adjust(top=0.94)
    plt.show()
    return fig, ax, x_data


def marker_graph_add(ax, marker_data, x_data, style):
    for n in range(3):
        ax[n].plot(x_data + 1, marker_data[x_data, n], style)


def make_interactive(multiple=True):
    mplcursors.cursor(multiple=multiple)
