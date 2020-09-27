if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import sys
    import distutils.util
    import matplotlib.pyplot as plt
    from biplane_kine.database import create_db
    from biplane_tasks.parameters import read_smoothing_exceptions
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.graph_utils import init_graphing
    from biplane_kine.database.c3d_helper import C3DSubjectEndpts
    from biplane_kine.misc.python_utils import partialclass
    from .smooth_marker import marker_plotter, figs_to_pdf

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
    db = create_db(params.db_dir, partialclass(C3DSubjectEndpts, labeled_base_dir=params.labeled_c3d_dir,
                                               filled_base_dir=params.filled_c3d_dir))

    # filter and plot
    trial_row = db.loc[params.trial_name]
    t = trial_row.Trial
    log.info('Filtering trial %s marker %s', t.trial_name, params.marker_name)
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
    init_graphing()
    figs = marker_plotter(t, params.marker_name, all_exceptions, db.attrs['dt'])
    if figs is None:
        sys.exit(1)
    plt.show()

    if bool(distutils.util.strtobool(params.print_to_file)):
        subj_dir = Path(params.output_dir) / trial_row.Subject_Name
        t_dir = subj_dir / trial_row.Trial_Name
        t_dir.mkdir(parents=True, exist_ok=True)
        figs_to_pdf(figs, t_dir, params.marker_name)
