"""Demonstrates the effect of smoothing for a specified trial/marker

This script compares both labeled (raw) and smoothed Vicon marker data against the marker position data obtain by
tracking the marker via biplane fluoroscopy. This is useful for determining the spatiotemporal syncing of the Vicon and
biplane fluoroscopy system, and to determine the effects of smoothing.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
accuracy_db_dir: Path to the directory containing marker data as tracked via biplane fluoroscopy
smoothing_exceptions: Path to a file containing smoothing exceptions for each trial/marker.
trial_name: Trial identifier for the marker to be smoothed.
marker_name: Marker to be smoothed.
labeled_c3d_dir: Path to directory where labeled C3D trial files are located.
filled_c3d_dir: Path to directory where filled C3D trial files are located.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any
from biplane_kine.database.c3d_helper import C3DTrial
from biplane_kine.database.vicon_accuracy import BiplaneMarkerTrialEndpts
from biplane_kine.vicon_biplane_diff import BiplaneViconSmoothDiff
import logging
log = logging.getLogger(__name__)


def add_c3d_helper(db_df: pd.DataFrame, vicon_labeled_path: Union[str, Path],
                   vicon_filled_path: Union[str, Path]) -> None:
    """Add C3D trial column to trial database."""
    def create_c3d_helper(row: pd.Series, labeled_path: Union[str, Path], filled_path: Union[str, Path]) -> C3DTrial:
        """Create a C3D trial from the specified trial database row."""
        return C3DTrial(str(Path(labeled_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')),
                        str(Path(filled_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')))

    db_df['C3D_Trial'] = db_df.apply(create_c3d_helper, axis=1, args=(vicon_labeled_path, vicon_filled_path))


def marker_accuracy_diff(biplane_trial: BiplaneMarkerTrialEndpts, c3d_trial: C3DTrial, marker: str,
                         marker_except: Dict[str, Any], dt: float, use_filled_portion: bool = True):
    """Create a BiplaneViconSmoothDiff object given the trial that contains Vicon marker data tracked via biplane
    fluoroscopy (biplane_trial) and the marker data as recorded via Vicon (c3d_trial)."""
    return BiplaneViconSmoothDiff(biplane_trial[marker], biplane_trial.vicon_endpts, biplane_trial.subject.f_t_v,
                                  c3d_trial.labeled[marker], c3d_trial.filled[marker], marker_except, dt,
                                  use_filled_portion)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    import distutils.util
    import matplotlib.pyplot as plt
    from biplane_tasks.parameters import smoothing_exceptions_for_marker
    from biplane_kine.database import create_db
    from biplane_kine.database.vicon_accuracy import BiplaneMarkerSubjectEndpts
    from biplane_kine.graphing.vicon_accuracy_plotters import ViconAccuracySmoothingPlotter
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from ..general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Demonstrates the effect of smoothing for a specified trial/marker', __package__,
                                     __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    db = create_db(params.accuracy_db_dir, BiplaneMarkerSubjectEndpts)
    add_c3d_helper(db, params.labeled_c3d_dir, params.filled_c3d_dir)

    # select trial
    trial_row = db.loc[params.trial_name]
    marker_exceptions = smoothing_exceptions_for_marker(params.smoothing_exceptions, params.trial_name,
                                                        params.marker_name)

    # compute differences
    try:
        raw_smoothed_diff = marker_accuracy_diff(trial_row['Biplane_Marker_Trial'], trial_row['C3D_Trial'],
                                                 params.marker_name, marker_exceptions, db.attrs['dt'],
                                                 bool(distutils.util.strtobool(params.use_filled_portion)))
    except InsufficientDataError as e:
        log.error('Insufficient data for trial {} marker {}: {}'.format(params.trial_name, params.marker_name, e))
        sys.exit(1)
    except DoNotUseMarkerError as e:
        log.error('Marker {} for trial {} should not be used: {}'.format(params.trial_name, params.marker_name, e))
        sys.exit(1)

    # plot
    init_graphing()
    plotter = ViconAccuracySmoothingPlotter(params.trial_name, params.marker_name, raw_smoothed_diff)
    plotter.plot()
    plt.show()
