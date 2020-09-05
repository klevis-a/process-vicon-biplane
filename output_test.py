import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from database import create_db
from parameters import Params
from database.dynamic_trial import DynamicTrial
import graphing.plotters as plotters
import graphing.graph_utils as graph


def trial_plotter(trial, dt, subj_dir):
    trial_pdf = subj_dir / (trial.trial_name + '.pdf')
    with PdfPages(trial_pdf) as pdf_file:
        for marker in DynamicTrial.MARKERS:
            if marker in trial.vicon_data_labeled.columns:
                marker_plotter = plotters.LabeledFilledMarkerPlotter(trial, marker, dt)
                fig = marker_plotter.plot()
                pdf_file.savefig(fig)
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
