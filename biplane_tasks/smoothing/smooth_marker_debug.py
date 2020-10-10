"""Smooth Vicon marker data using Kalman smoothing and optionally creates a debug PDF record

This script smooths a specific Vicon marker (from an individual trial) and optionally creates a debug PDF record of the
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

from biplane_kine.graphing.smoothing_plotters import SmoothingDebugPlotter
from biplane_kine.smoothing.kalman_filtering import FilterStep
from typing import Union, Sequence
import numpy as np
import logging

log = logging.getLogger(__name__)


class SmoothingDebugPlotterProxy(SmoothingDebugPlotter):
    """A proxy class for the SmoothingDebugPlotter that imitates the constructor signature of SmoothingOutputPlotter but
    creates a SmoothingDebugPlotter."""
    def __init__(self, trial_name: str, marker_name: str, raw: FilterStep, filled: FilterStep, filtered: FilterStep,
                 smoothed: FilterStep, vicon_endpts: Union[np.ndarray, Sequence]):
        super().__init__(trial_name, marker_name, raw, filtered, smoothed, vicon_endpts)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    import distutils.util
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from biplane_tasks.parameters import smoothing_exceptions_for_marker
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from .smooth_marker import marker_plotter
    from ..general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Smooth Vicon marker data using Kalman smoothing and optionally creates a debug '
                                     'PDF record', __package__, __file__))
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
        marker_plotter(sel_trial, params.marker_name, marker_exceptions, db.attrs['dt'], SmoothingDebugPlotterProxy,
                       subject_dir)
    except InsufficientDataError as e:
        log.error('Insufficient data for trial {} marker {}: {}'.format(params.trial_name, params.marker_name, e))
        sys.exit(1)
    except DoNotUseMarkerError as e:
        log.error('Marker {} for trial {} should not be used: {}'.format(params.trial_name, params.marker_name, e))
        sys.exit(1)
