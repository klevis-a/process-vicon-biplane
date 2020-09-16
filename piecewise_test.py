import os
from pathlib import Path
import sys
import distutils.util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params
import graphing.graph_utils as graph
from graphing.plotters import SmoothingOutputPlotter
from smoothing.kf_filtering_helpers import kf_filter_marker_all, post_process_raw, kf_filter_marker_piecewise, \
    combine_pieces

# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)

# filter
trial_row = db.loc[params.trial_name]
trial = trial_row.Trial
raw, filled = post_process_raw(trial, params.marker_name, dt=db.attrs['dt'])
filtered, smoothed = kf_filter_marker_all(trial, params.marker_name, dt=db.attrs['dt'])
filtered_pieces, smoothed_pieces = kf_filter_marker_piecewise(trial, params.marker_name, dt=db.attrs['dt'])
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