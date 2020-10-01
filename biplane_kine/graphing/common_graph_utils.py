import matplotlib
import mplcursors
from matplotlib import rcParams


def init_graphing():
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    matplotlib.use('Qt5Agg')


def make_interactive(multiple=True):
    mplcursors.cursor(multiple=multiple)
