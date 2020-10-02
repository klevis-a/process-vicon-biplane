import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from biplane_kine.database.db_common import MARKERS
from biplane_kine.graphing.smoothing_plotters import SmoothingOutputPlotter
from biplane_kine.smoothing.kf_filtering_helpers import (piecewise_filter_with_exception, InsufficientDataError,
                                                         DoNotUseMarkerError)
import logging
from ..parameters import marker_smoothing_exceptions

log = logging.getLogger(__name__)


def create_and_save_error_figure(msg, pdf_file):
    temp_fig = plt.figure()
    temp_fig.suptitle(msg, fontsize=11, fontweight='bold')
    pdf_file.savefig(temp_fig)
    temp_fig.clf()
    plt.close(temp_fig)


def trial_plotter(trial, dt, subj_dir, all_smoothing_except):
    log.info('Smoothing trial %s', trial.trial_name)
    trial_dir = subj_dir / trial.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_pdf_file = subj_dir / (trial.trial_name + '.pdf')
    with PdfPages(trial_pdf_file) as trial_pdf:
        for marker in MARKERS:
            if marker in trial.vicon_csv_data_labeled.columns:
                log.info('Smoothing marker %s', marker)
                marker_pdf_file = trial_dir / (marker + '.pdf')
                try:
                    marker_exceptions = marker_smoothing_exceptions(all_smoothing_except, trial.trial_name, marker)
                    raw, filled, filtered, smoothed = \
                        piecewise_filter_with_exception(marker_exceptions, trial.labeled[marker], trial.filled[marker],
                                                        dt)
                except InsufficientDataError as e:
                    create_and_save_error_figure(marker + ' Insufficient Data', trial_pdf)
                    log.warning('Skipping marker {} for trial {} because: {}'.format(marker, trial.trial_name, e))
                    continue
                except DoNotUseMarkerError as e:
                    create_and_save_error_figure(marker + ' SHOULD NOT USE', trial_pdf)
                    log.warning('Skipping marker {} for trial {} because: {}'.format(marker, trial.trial_name, e))
                    continue
                marker_plotter = SmoothingOutputPlotter(trial.trial_name, marker, raw, filled, filtered, smoothed,
                                                        trial.vicon_endpts)
                figs = marker_plotter.plot()
                with PdfPages(marker_pdf_file) as marker_pdf:
                    for (fig_num, fig) in enumerate(figs):
                        marker_pdf.savefig(fig)
                        if fig_num in [0, 1]:
                            trial_pdf.savefig(fig)
                        fig.clf()
                        plt.close(fig)
            else:
                log.warning('Marker %s missing', marker)
                create_and_save_error_figure(marker, trial_pdf)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    from biplane_kine.database import create_db
    from biplane_kine.database.vicon_csv import ViconCsvSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from ..parameters import read_smoothing_exceptions
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.vicon_csv_dir, ViconCsvSubject)
    init_graphing()
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)

    # create plots
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Smoothing subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for t in subject_df['Trial']:
            trial_plotter(t, db.attrs['dt'], subject_dir, all_exceptions)
