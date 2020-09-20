import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params, read_smoothing_exceptions, marker_smoothing_exceptions
from database.dynamic_trial import DynamicTrial
from graphing.plotters import SmoothingOutputPlotter
import graphing.graph_utils as graph
from smoothing.kf_filtering_helpers import post_process_raw, kf_filter_marker_piecewise, combine_pieces, \
    InsufficientDataError
import logging
from logging.config import fileConfig

fileConfig('logging_config.ini', disable_existing_loggers=False)
log = logging.getLogger('kf_smoothing')


def trial_plotter(trial, dt, subj_dir, all_smoothing_except):
    log.info('Smoothing trial %s', trial.trial_name)
    trial_dir = subj_dir / trial.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_pdf_file = subj_dir / (trial.trial_name + '.pdf')
    with PdfPages(trial_pdf_file) as trial_pdf:
        for marker in DynamicTrial.MARKERS:
            if marker in trial.vicon_data_labeled.columns:
                log.info('Smoothing marker %s', marker)
                marker_pdf_file = trial_dir / (marker + '.pdf')
                try:
                    marker_exceptions = marker_smoothing_exceptions(all_smoothing_except, trial.trial_name, marker)
                    raw, filled = post_process_raw(trial, marker, dt=dt)
                    filtered_pieces, smoothed_pieces = kf_filter_marker_piecewise(trial, marker, dt=dt,
                                                                                  **marker_exceptions)
                    filtered = combine_pieces(filtered_pieces)
                    smoothed = combine_pieces(smoothed_pieces)
                except InsufficientDataError:
                    temp_fig = plt.figure()
                    temp_fig.suptitle(marker + ' Insufficient Data', fontsize=11, fontweight='bold')
                    trial_pdf.savefig(temp_fig)
                    temp_fig.clf()
                    plt.close(temp_fig)
                    log.warning('Skipping marker %s for trial %s because there is insufficient data to filter',
                                marker, trial.trial_name)
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
                temp_fig = plt.figure()
                temp_fig.suptitle(marker, fontsize=11, fontweight='bold')
                trial_pdf.savefig(temp_fig)
                temp_fig.clf()
                plt.close(temp_fig)


# ready db
params = Params.get_params(sys.argv[1])
root_path = Path(params.output_dir)
db, anthro = create_db(params.db_dir)
graph.init_graphing()
all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)

# create plots
for subject_name, subject_df in db.groupby('Subject'):
    log.info('Smoothing subject %s', subject_name)
    subject_dir = (root_path / subject_name)
    subject_dir.mkdir(parents=True, exist_ok=True)
    subject_df['Trial'].apply(trial_plotter, args=(db.attrs['dt'], subject_dir, all_exceptions))
