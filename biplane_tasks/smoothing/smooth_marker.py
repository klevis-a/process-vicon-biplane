import os
import numpy as np
import distutils.util
from matplotlib.backends.backend_pdf import PdfPages
from biplane_tasks.parameters import marker_smoothing_exceptions
from biplane_kine.smoothing.kf_filtering_helpers import post_process_raw, kf_filter_marker_piecewise, combine_pieces
import logging
log = logging.getLogger(__name__)


def marker_plotter(trial, marker_name, all_except, dt, plotter_cls):
    marker_exceptions = marker_smoothing_exceptions(all_except, trial.trial_name, marker_name)
    should_use = bool(distutils.util.strtobool(marker_exceptions.get('use_marker', 'True')))
    if not should_use:
        log.warning('Skipping marker because it is labeled as DO NOT USE.')
        return None
    smoothing_params = marker_exceptions.get('smoothing_params', {})
    frame_ignores = np.asarray(marker_exceptions.get('frame_ignores', []))

    # ignore frames
    if frame_ignores.size > 0:
        trial.marker_data_labeled(marker_name)[frame_ignores - 1, :] = np.nan

    raw, filled = post_process_raw(trial, marker_name, dt)
    filtered_pieces, smoothed_pieces = kf_filter_marker_piecewise(trial, marker_name, dt, **smoothing_params)
    filtered = combine_pieces(filtered_pieces)
    smoothed = combine_pieces(smoothed_pieces)

    plotter = plotter_cls(trial.trial_name, marker_name, raw, filled, filtered, smoothed, trial.vicon_endpts)

    return plotter.plot()


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
    import matplotlib.pyplot as plt
    from biplane_kine.database import create_db
    from biplane_kine.database.dynamic_subject import DynamicSubject
    from biplane_kine.graphing.smoothing_plotters import SmoothingOutputPlotter
    from biplane_tasks.parameters import read_smoothing_exceptions
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
    db = create_db(params.db_dir, DynamicSubject)

    # filter and plot
    trial_row = db.loc[params.trial_name]
    t = trial_row.Trial
    log.info('Filtering trial %s marker %s', t.trial_name, params.marker_name)
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
    init_graphing()
    figs = marker_plotter(t, params.marker_name, all_exceptions, db.attrs['dt'], SmoothingOutputPlotter)
    if figs is None:
        sys.exit(1)
    plt.show()

    if bool(distutils.util.strtobool(params.print_to_file)):
        subj_dir = Path(params.output_dir) / trial_row.Subject_Name
        t_dir = subj_dir / trial_row.Trial_Name
        t_dir.mkdir(parents=True, exist_ok=True)
        figs_to_pdf(figs, t_dir, params.marker_name)
