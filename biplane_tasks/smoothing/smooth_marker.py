"""Smooth Vicon marker data using Kalman smoothing and optionally creates a PDF record

This script smooths a specific Vicon marker (from an individual trial) and optionally creates a PDF record of the
smoothing effects. In order to not override existing files, an increasing integer is appended to the PDF file name.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to the directory where PDF records for the marker will be output.
smoothing_exceptions: Path to a file containing smoothing exceptions for each trial/marker.
trial_name: Trial identifier for the marker to be smoothed.
marker_name: Marker to be smoothed.
print_to_file: Whether to print the resulting graphs to a PDF file (can be case insensitive y/n, yes/no, true/false,
               1/0)
"""

import os
import matplotlib.figure
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Type, Union, List
from matplotlib.backends.backend_pdf import PdfPages
from biplane_kine.database.biplane_vicon_db import ViconCsvTrial
from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception
import logging
log = logging.getLogger(__name__)


def marker_plotter(trial: ViconCsvTrial, marker_name: str, marker_except: Dict[str, Any], dt: float, plotter_cls: Type,
                   subj_dir: Union[Path, None] = None) -> None:
    """Plot the specified marker (marker_name) specified in trial using the plotter specified by plotter_cls."""
    raw, filled, filtered, smoothed = \
        piecewise_filter_with_exception(marker_except, trial.labeled[marker_name], trial.filled[marker_name], dt)

    plotter = plotter_cls(trial.trial_name, marker_name, raw, filled, filtered, smoothed, trial.vicon_endpts)
    figs = plotter.plot()
    plt.show()

    if subj_dir:
        trial_dir = subj_dir / trial.trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        figs_to_pdf(figs, trial_dir, marker_name)


def figs_to_pdf(figures: List[matplotlib.figure.Figure], trial_dir: Path, marker_name: str):
    """Record figures to a PDF file that does not override any existing PDF records."""

    # make sure to not override existing file, makes comparisons easier
    marker_pdf_file = str(trial_dir / (marker_name + '{}.pdf'))
    counter = 0
    while os.path.isfile(marker_pdf_file.format(counter)):
        counter += 1
    marker_pdf_file = marker_pdf_file.format(counter)

    with PdfPages(marker_pdf_file) as marker_pdf:
        for fig in figures:
            marker_pdf.savefig(fig)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    import distutils.util
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.graphing.smoothing_plotters import SmoothingOutputPlotter
    from biplane_tasks.parameters import smoothing_exceptions_for_marker
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.misc.arg_parser import mod_arg_parser
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Smooth Vicon marker data using Kalman smoothing and optionally creates a PDF '
                                     'record', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, ViconCsvSubject)

    # select trial
    trial_row = db.loc[params.trial_name]
    sel_trial = trial_row.Trial
    log.info('Filtering trial %s marker %s', params.trial_name, params.marker_name)

    # filter and plot
    marker_exceptions = smoothing_exceptions_for_marker(params.smoothing_exceptions, params.trial_name,
                                                        params.marker_name)
    init_graphing()
    subject_dir = Path(params.output_dir) / trial_row.Subject_Name if \
        bool(distutils.util.strtobool(params.print_to_file)) else None
    try:
        marker_plotter(sel_trial, params.marker_name, marker_exceptions, db.attrs['dt'], SmoothingOutputPlotter,
                       subject_dir)
    except InsufficientDataError as e:
        log.error('Insufficient data for trial {} marker {}: {}'.format(params.trial_name, params.marker_name, e))
        sys.exit(1)
    except DoNotUseMarkerError as e:
        log.error('Marker {} for trial {} should not be used: {}'.format(params.trial_name, params.marker_name, e))
        sys.exit(1)
