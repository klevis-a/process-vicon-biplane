import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params
from database.dynamic_trial import DynamicTrial
from graphing.plotters import SmoothingOutputPlotter
import graphing.graph_utils as graph
from smoothing.kf_filtering_helpers import kf_filter_marker


def trial_plotter(trial, dt, subj_dir):
    trial_dir = subj_dir / trial.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    print('Smoothing trial {}'.format(trial.trial_name))
    for marker in DynamicTrial.MARKERS:
        if marker in trial.vicon_data_labeled.columns:
            marker_pdf = trial_dir / (marker + '.pdf')
            with PdfPages(marker_pdf) as pdf_file:
                raw, filled, filtered, smoothed = kf_filter_marker(trial, marker, dt=dt)
                marker_plotter = SmoothingOutputPlotter(trial.trial_name, marker, raw, filled, filtered, smoothed,
                                                        trial.vicon_endpts)
                figs = marker_plotter.plot()
                for fig in figs:
                    pdf_file.savefig(fig)
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
