"""Smooth Vicon marker data (obtained directly from C3D file) using Kalman smoothing and optionally creates a PDF record

This script smooths a specific Vicon marker (from an individual trial) and optionally creates a PDF record of the
smoothing effects. The data is obtained directly from the C3D file, rather than a CSV export of it. In order to not
override existing files, an increasing integer is appended to the PDF file name.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data, this is used to retrieve the frame
                      indices (endpoints) of the Vicon trial that correspond to the endpoints of the reciprocal biplane
                      fluoroscopy trial
output_dir: Path to the directory where PDF records for the marker will be output.
smoothing_exceptions: Path to a file containing smoothing exceptions for each trial/marker.
trial_name: Trial identifier for the marker to be smoothed.
marker_name: Marker to be smoothed.
print_to_file: Whether to print the resulting graphs to a PDF file (can be case insensitive y/n, yes/no, true/false,
               1/0).
labeled_c3d_dir: Path to directory where labeled C3D trial files are located.
filled_c3d_dir: Path to directory where filled C3D trial files are located.
"""

import numpy as np
from lazy import lazy
from biplane_kine.database.c3d_helper import C3DTrialEndpts, C3DHelper
from biplane_kine.misc.python_utils import NestedContainer
import logging

log = logging.getLogger(__name__)


def c3d_get_item_method_non_hom(c3d_helper: C3DHelper, item: str) -> np.ndarray:
    """Get marker data from c3d_helper and remove the homogeneous fourth dimension."""
    return c3d_helper[item][:, :3]


class C3DTrialEndptsNonHom(C3DTrialEndpts):
    """A proxy for the C3DTrialEndpts that returns non-homogeneous marker data (N, 3)."""
    @lazy
    def labeled(self) -> NestedContainer:
        """Descriptor that allows marker indexed ([marker_name]) access to labeled C3D data. The indexed access returns
         a (n, 3) numpy array view associated with marker_name."""
        return NestedContainer(self.labeled_c3d, c3d_get_item_method_non_hom)

    @lazy
    def filled(self) -> NestedContainer:
        """Descriptor that allows marker indexed ([marker_name]) access to filled C3D data. The indexed access returns
         a (n, 3) numpy array view associated with marker_name."""
        return NestedContainer(self.filled_c3d, c3d_get_item_method_non_hom)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import sys
    import distutils.util
    from typing import Union
    from biplane_kine.database import create_db
    from biplane_kine.graphing.smoothing_plotters import SmoothingOutputPlotter
    from biplane_tasks.parameters import smoothing_exceptions_for_marker
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.database.c3d_helper import C3DSubjectEndpts
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from .smooth_marker import marker_plotter
    from ..general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Time database creation', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    class C3DSubjectEndptsPrePop(C3DSubjectEndpts):
        """Proxy for C3DSubjectEndpts where the labeled_base_dir and filled_base_dir are pre-populated."""
        def __init__(self, subj_dir: Union[str, Path]):
            super().__init__(subj_dir, labeled_base_dir=params.labeled_c3d_dir, filled_base_dir=params.filled_c3d_dir,
                             c3d_trial_cls=C3DTrialEndptsNonHom)


    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, C3DSubjectEndptsPrePop)

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
