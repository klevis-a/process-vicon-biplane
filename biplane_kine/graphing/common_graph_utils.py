"""This module provides utilities that will be broadly used by code that performs plotting."""

import matplotlib
import mplcursors
from matplotlib import rcParams


def init_graphing() -> None:
    """Specify the default font and backend for Matplotlib."""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    matplotlib.use('TkAgg')


def make_interactive(multiple: bool = True) -> None:
    """Enable interaction with the current Matplotlib figure."""
    mplcursors.cursor(multiple=multiple)
