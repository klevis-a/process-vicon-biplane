import matplotlib
import mplcursors
from matplotlib import rcParams


class MplStyle:
    def __init__(self, color, ls='-', lw=2, marker='', ms=4):
        self.color = color
        self.ls = ls
        self.lw = lw
        self.marker = marker
        self.ms = ms


def init_graphing():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    matplotlib.use('Qt5Agg')


def make_interactive(multiple=True):
    mplcursors.cursor(multiple=multiple)
