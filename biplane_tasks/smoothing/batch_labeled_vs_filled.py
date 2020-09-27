import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from biplane_kine.database.db_common import MARKERS
from biplane_kine.graphing.plotters import LabeledFilledMarkerPlotter
import logging

log = logging.getLogger(__name__)


def trial_plotter(trial, subj_dir):
    log.info('Outputting trial %s', trial.trial_name)
    trial_pdf_file = subj_dir / (trial.trial_name + '.pdf')
    with PdfPages(trial_pdf_file) as trial_pdf:
        for marker in MARKERS:
            if marker in trial.vicon_data_labeled.columns:
                log.info('Outputting marker %s', marker)
                marker_plotter = LabeledFilledMarkerPlotter(trial, marker)
                figs = marker_plotter.plot()
                for fig in figs:
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


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    from biplane_kine.database import create_db
    from biplane_kine.database.dynamic_subject import DynamicSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.graph_utils import init_graphing
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
    init_graphing()

    # create plots
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_df['Trial'].apply(trial_plotter, args=[subject_dir])
