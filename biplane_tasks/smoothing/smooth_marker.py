import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception
import logging
log = logging.getLogger(__name__)


def marker_plotter(trial, marker_name, marker_except, dt, plotter_cls, subj_dir=None):
    raw, filled, filtered, smoothed = \
        piecewise_filter_with_exception(marker_except, trial.labeled[marker_name], trial.filled[marker_name], dt)

    plotter = plotter_cls(trial.trial_name, marker_name, raw, filled, filtered, smoothed, trial.vicon_endpts)
    figs = plotter.plot()
    plt.show()

    if subj_dir:
        trial_dir = subj_dir / trial.trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        figs_to_pdf(figs, trial_dir, marker_name)


def figs_to_pdf(figures, trial_dir, marker_name):
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
    from pathlib import Path
    import distutils.util
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.graphing.smoothing_plotters import SmoothingOutputPlotter
    from biplane_tasks.parameters import smoothing_exceptions_for_marker
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
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
