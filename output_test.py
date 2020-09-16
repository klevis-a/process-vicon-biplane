import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params
from database.dynamic_trial import DynamicTrial
from graphing.plotters import SmoothingOutputPlotter
import graphing.graph_utils as graph
from smoothing.kf_filtering_helpers import kf_filter_marker_all, InsufficientDataError


def trial_plotter(trial, dt, subj_dir):
    trial_dir = subj_dir / trial.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    print('Smoothing trial {}'.format(trial.trial_name))
    trial_pdf_file = trial_dir / (trial.trial_name + '.pdf')
    with PdfPages(trial_pdf_file) as trial_pdf:
        for marker in DynamicTrial.MARKERS:
            if marker in trial.vicon_data_labeled.columns:
                marker_pdf_file = trial_dir / (marker + '.pdf')
                try:
                    raw, filled, filtered, smoothed = kf_filter_marker_all(trial, marker, dt=dt)
                except InsufficientDataError:
                    print('Skipping marker {} for trial {} because there is insufficient data to filter.'
                          .format(marker, trial.trial_name))
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


# ready db
params = Params.get_params(sys.argv[1])
root_path = Path(params.output_dir)
db, anthro = create_db(params.db_dir)
graph.init_graphing()

# create plots
for subject_name, subject_df in db.groupby('Subject'):
    subject_dir = (root_path / subject_name)
    subject_dir.mkdir(parents=True, exist_ok=True)
    subject_df['Trial'].apply(trial_plotter, args=(db.attrs['dt'], subject_dir))
