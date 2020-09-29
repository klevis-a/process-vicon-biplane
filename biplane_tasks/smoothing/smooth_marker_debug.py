from .smooth_marker import marker_plotter, figs_to_pdf
from biplane_kine.graphing.smoothing_plotters import SmoothingDebugPlotter


class SmoothingDebugPlotterProxy(SmoothingDebugPlotter):
    def __init__(self, trial_name, marker_name, raw, filled, filtered, smoothed, vicon_endpts):
        super().__init__(trial_name, marker_name, raw, filtered, smoothed, vicon_endpts)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    import distutils.util
    import matplotlib.pyplot as plt
    from biplane_kine.database import create_db
    from biplane_kine.database.dynamic_subject import DynamicSubject
    from biplane_tasks.parameters import read_smoothing_exceptions
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
    root_path = Path(params.output_dir)
    db = create_db(params.db_dir, DynamicSubject)

    # filter and plot
    trial_row = db.loc[params.trial_name]
    t = trial_row.Trial
    log.info('Filtering trial %s marker %s', t.trial_name, params.marker_name)
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
    init_graphing()
    figs = marker_plotter(t, params.marker_name, all_exceptions, db.attrs['dt'], SmoothingDebugPlotterProxy)
    if figs is None:
        sys.exit(1)
    plt.show()

    if bool(distutils.util.strtobool(params.print_to_file)):
        subj_dir = Path(params.output_dir) / trial_row.Subject_Name
        t_dir = subj_dir / trial_row.Trial_Name
        t_dir.mkdir(parents=True, exist_ok=True)
        figs_to_pdf(figs, t_dir, params.marker_name)
