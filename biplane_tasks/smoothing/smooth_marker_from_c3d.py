from lazy import lazy
from biplane_kine.database.c3d_helper import C3DTrialEndpts
from biplane_kine.misc.python_utils import NestedContainer
import logging

log = logging.getLogger(__name__)


def c3d_get_item_method_non_hom(c3d_helper, item):
    return c3d_helper[item][:, :3]


class C3DTrialEndptsNonHom(C3DTrialEndpts):
    @lazy
    def labeled(self):
        return NestedContainer(self.labeled_c3d, c3d_get_item_method_non_hom)

    @lazy
    def filled(self):
        return NestedContainer(self.filled_c3d, c3d_get_item_method_non_hom)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import sys
    import distutils.util
    from biplane_kine.database import create_db
    from biplane_kine.graphing.smoothing_plotters import SmoothingOutputPlotter
    from biplane_tasks.parameters import smoothing_exceptions_for_marker
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.database.c3d_helper import C3DSubjectEndpts
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from .smooth_marker import marker_plotter
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    class C3DSubjectEndptsPrePop(C3DSubjectEndpts):
        def __init__(self, subj_dir):
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
