"""Graph torso position and euler angles derived from labeled, filled, and smoothed marker data for every trial in
the biplane/Vicon filesystem-based database

The Visual3D torso coordinate system definition is used to facilitate comparison with previoulsy generated data.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where PDF file summaries should be created
"""

import numpy as np
from biplane_kine.kinematics.cs import change_cs, ht_inv
from biplane_kine.kinematics.euler_angles import zxy_intrinsic
from biplane_kine.graphing.kine_plotters import RawSmoothedKineTorsoPlotter
import logging

log = logging.getLogger(__name__)


def trial_plotter(trial_row):
    log.info('Processing trial %s', trial_row['Trial_Name'])

    def process_trial(trial):
        torso_truncated = trial.torso_fluoro[trial.vicon_endpts[0]:trial.vicon_endpts[1]]
        torso_intrinsic = change_cs(ht_inv(torso_truncated[1]), torso_truncated)
        torso_pos = torso_intrinsic[:, :3, 3]
        torso_eul = np.rad2deg(zxy_intrinsic(torso_intrinsic))
        return torso_pos, torso_eul

    torso_pos_smoothed, torso_eul_smoothed = process_trial(trial_row['Smoothed_Trial'])
    torso_pos_labeled, torso_eul_labeled = process_trial(trial_row['Labeled_Trial'])
    torso_pos_filled, torso_eul_filled = process_trial(trial_row['Filled_Trial'])

    # graph
    frame_nums = np.arange(trial_row['Smoothed_Trial'].vicon_endpts[0], trial_row['Smoothed_Trial'].vicon_endpts[1]) + 1
    return RawSmoothedKineTorsoPlotter(trial_row['Trial_Name'], torso_pos_labeled, torso_eul_labeled, torso_pos_filled,
                                       torso_eul_filled, torso_pos_smoothed, torso_eul_smoothed, frame_nums)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import pandas as pd
    from typing import Union
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ..general.arg_parser import mod_arg_parser
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.database import create_db, anthro_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject, BiplaneViconSubjectV3D, BiplaneViconTrial
    from biplane_kine.misc.json_utils import Params
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Compare torso CS as defined via Visual3D and ISB.', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    anthro = anthro_db(params.biplane_vicon_db_dir)

    def armpit_thickness(subj_name):
        return anthro.loc[subj_name, 'Armpit_Thickness']

    class BiplaneViconTrialLabeled(BiplaneViconTrial):
        def __init__(self, trial_dir: Union[str, Path], subject: BiplaneViconSubject, nan_missing_markers: bool = True,
                     **kwargs):
            super().__init__(trial_dir, subject, nan_missing_markers, **kwargs)
            self.torso_source = 'labeled'


    class BiplaneViconTrialFilled(BiplaneViconTrial):
        def __init__(self, trial_dir: Union[str, Path], subject: BiplaneViconSubject, nan_missing_markers: bool = True,
                     **kwargs):
            super().__init__(trial_dir, subject, nan_missing_markers, **kwargs)
            self.torso_source = 'filled'

    db_smoothed = create_db(params.biplane_vicon_db_dir, BiplaneViconSubjectV3D, armpit_thickness=armpit_thickness)
    db_labeled = create_db(params.biplane_vicon_db_dir, BiplaneViconSubjectV3D, armpit_thickness=armpit_thickness,
                           trial_class=BiplaneViconTrialLabeled)
    db_filled = create_db(params.biplane_vicon_db_dir, BiplaneViconSubjectV3D, armpit_thickness=armpit_thickness,
                          trial_class=BiplaneViconTrialFilled)
    db = pd.DataFrame({'Trial_Name': db_smoothed['Trial_Name'], 'Subject_Name': db_smoothed['Subject_Name'],
                       'Labeled_Trial': db_labeled['Trial'], 'Filled_Trial': db_filled['Trial'],
                       'Smoothed_Trial': db_smoothed['Trial']}, index=db_smoothed['Trial_Name'])

    db['Plotter'] = db.apply(trial_plotter, axis=1)

    # create plots
    init_graphing()
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting kinematics for subject %s', subject_name)
        subject_pdf_file = root_path / (subject_name + '.pdf')
        with PdfPages(subject_pdf_file) as subject_pdf:
            for plotter in subject_df['Plotter']:
                figs = plotter.plot()
                subject_pdf.savefig(figs[3])
                subject_pdf.savefig(figs[2])
                for fig in figs:
                    fig.clf()
                    plt.close(fig)
