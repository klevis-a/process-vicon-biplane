import os
from pathlib import Path
import sys
import distutils.util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params, marker_smoothing_exceptions, read_smoothing_exceptions
import graphing.graph_utils as graph
from graphing.plotters import SmoothingOutputPlotter
from smoothing.kf_filtering_helpers import post_process_raw, kf_filter_marker_piecewise, combine_pieces
from database.dynamic_trial import DynamicTrial
import logging
from logging.config import fileConfig

fileConfig('logging_config.ini', disable_existing_loggers=False)
log = logging.getLogger('kf_smoothing')


def marker_data_labeled(self, marker_name):
    return self.c3d_helper_labeled.data_for_marker(marker_name)


def marker_data_filled(self, marker_name):
    return self.c3d_helper_filled.data_for_marker(marker_name)


def new_c3d_path(row, labeled_base_path, filled_base_path):
    row['Trial'].c3d_file_labeled = labeled_base_path / row['Subject'] / (row['Trial_Name'] + '.c3d')
    row['Trial'].c3d_file_filled = filled_base_path / row['Subject'] / (row['Trial_Name'] + '.c3d')


# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)

# modify DynamicTrial so it uses c3d
delattr(DynamicTrial, 'marker_data_labeled_df')
delattr(DynamicTrial, 'marker_data_filled_df')
DynamicTrial.marker_data_labeled = marker_data_labeled
DynamicTrial.marker_data_filled = marker_data_filled
db.apply(new_c3d_path, axis=1, args=(Path(params.labeled_c3d_dir), Path(params.filled_c3d_dir)))


# filter
trial_row = db.loc[params.trial_name]
trial = trial_row.Trial
log.info('Filtering trial %s marker %s', trial.trial_name, params.marker_name)
all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
marker_exceptions = marker_smoothing_exceptions(all_exceptions, params.trial_name, params.marker_name)

raw, filled = post_process_raw(trial, params.marker_name, dt=db.attrs['dt'])
filtered_pieces, smoothed_pieces = kf_filter_marker_piecewise(trial, params.marker_name, dt=db.attrs['dt'],
                                                              **marker_exceptions)
filtered = combine_pieces(filtered_pieces)
smoothed = combine_pieces(smoothed_pieces)

# graphing
graph.init_graphing()
marker_plotter = SmoothingOutputPlotter(trial.trial_name, params.marker_name, raw, filled, filtered, smoothed,
                                        trial.vicon_endpts)
figs = marker_plotter.plot()
plt.show()

if bool(distutils.util.strtobool(params.print_to_file)):
    subj_dir = Path(params.output_dir) / trial_row.Subject
    trial_dir = subj_dir / trial_row.Trial_Name
    trial_dir.mkdir(parents=True, exist_ok=True)
    # make sure to not override existing file, makes comparisons easier
    marker_pdf_file = str(trial_dir / (params.marker_name + '{}.pdf'))
    counter = 0
    while os.path.isfile(marker_pdf_file.format(counter)):
        counter += 1
    marker_pdf_file = marker_pdf_file.format(counter)

    with PdfPages(marker_pdf_file) as marker_pdf:
        for (fig_num, fig) in enumerate(figs):
            marker_pdf.savefig(fig)
