from pathlib import Path
from biplane_kine.database.c3d_helper import C3DTrial
from .biplane_vicon_diff import BiplaneViconSmoothDiff
import logging
log = logging.getLogger(__name__)


def add_c3d_helper(db_df, vicon_labeled_path, vicon_filled_path):
    def create_c3d_helper(row, labeled_path, filled_path):
        return C3DTrial(str(Path(labeled_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')),
                        str(Path(filled_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')))

    db_df['C3D_Trial'] = db_df.apply(create_c3d_helper, axis=1, args=[vicon_labeled_path, vicon_filled_path])


def marker_accuracy_diff(biplane_trial, c3d_trial, marker, marker_except, dt, use_filled_portion=True):
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
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
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
