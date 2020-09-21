import os
from pathlib import Path
import sys
import numpy as np
import distutils.util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params, marker_smoothing_exceptions, read_smoothing_exceptions
import graphing.graph_utils as graph
from graphing.plotters import SmoothingOutputPlotter
from smoothing.kf_filtering_helpers import post_process_raw, kf_filter_marker_piecewise, combine_pieces
import logging
from logging.config import fileConfig

fileConfig('logging_config.ini', disable_existing_loggers=False)
log = logging.getLogger('kf_smoothing')

# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)

# filter
trial_row = db.loc[params.trial_name]
trial = trial_row.Trial
log.info('Filtering trial %s marker %s', trial.trial_name, params.marker_name)
all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
marker_exceptions = marker_smoothing_exceptions(all_exceptions, params.trial_name, params.marker_name)
should_use = bool(distutils.util.strtobool(marker_exceptions.get('use_marker', 'True')))
if not bool(distutils.util.strtobool(marker_exceptions.get('use_marker', 'True'))):
    log.warning('Skipping marker because it is labeled as DO NOT USE.')
    sys.exit(1)
smoothing_params = marker_exceptions.get('smoothing_params', {})
frame_ignores = np.asarray(marker_exceptions.get('frame_ignores', []))

# ignore frames
if frame_ignores.size > 0:
    trial.marker_data_labeled(params.marker_name)[frame_ignores - 1, :] = np.nan

raw, filled = post_process_raw(trial, params.marker_name, dt=db.attrs['dt'])
filtered_pieces, smoothed_pieces = kf_filter_marker_piecewise(trial, params.marker_name, db.attrs['dt'],
                                                              **smoothing_params)
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
