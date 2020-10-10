"""Batch create PDF records of raw (labeled) vs filled Vicon marker data

This script iterates over the Vicon/biplane fluoroscopy filesystem-based database and creates PDF records of the
difference between raw (labeled) vs filled Vicon marker data.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to the directory where PDF records for trial will be output.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from biplane_kine.database.db_common import MARKERS
from biplane_kine.database.biplane_vicon_db import ViconCsvTrial
from biplane_kine.graphing.smoothing_plotters import LabeledFilledMarkerPlotter
from .batch_kf_smoothing import create_and_save_error_figure
import logging

log = logging.getLogger(__name__)


def trial_plotter(trial: ViconCsvTrial, subj_dir: Path) -> None:
    log.info('Outputting trial %s', trial.trial_name)
    trial_pdf_file = subj_dir / (trial.trial_name + '.pdf')
    with PdfPages(trial_pdf_file) as trial_pdf:
        for marker in MARKERS:
            if marker in trial.vicon_csv_data_labeled.columns:
                log.info('Outputting marker %s', marker)
                marker_plotter = LabeledFilledMarkerPlotter(trial.trial_name, marker, trial.labeled[marker],
                                                            trial.filled[marker], trial.vicon_endpts)
                figs = marker_plotter.plot()
                for fig in figs:
                    trial_pdf.savefig(fig)
                    fig.clf()
                    plt.close(fig)
            else:
                create_and_save_error_figure(marker, trial_pdf)
                log.warning('Marker %s missing', marker)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from ..general.arg_parser import mod_arg_parser
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch create PDF records of raw (labeled) vs filled Vicon marker data',
                                     __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, ViconCsvSubject)
    init_graphing()

    # create plots
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_df['Trial'].apply(trial_plotter, args=[subject_dir])
