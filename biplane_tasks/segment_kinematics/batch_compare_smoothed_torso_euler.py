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
from biplane_kine.kinematics.kine_trajectory import compute_trajectory
from biplane_kine.kinematics.segments import StaticTorsoSegment
import logging

log = logging.getLogger(__name__)


def trial_plotter(trial_row):
    log.info('Processing trial %s', trial_row['Trial_Name'])

    def process_trial(trial, torso_source, base_frame_inv=None):
        tracking_markers = np.stack([getattr(trial, torso_source)[marker]
                                     for marker in StaticTorsoSegment.TRACKING_MARKERS], 0)
        torso_vicon = compute_trajectory(trial.subject.torso.static_markers_intrinsic, tracking_markers)
        torso_fluoro = change_cs(trial.subject.f_t_v, torso_vicon)
        torso_truncated = torso_fluoro[trial.vicon_endpts[0]:trial.vicon_endpts[1]]
        present_frames = np.nonzero(~np.any(np.isnan(torso_truncated), (-2, -1)))[0]
        if present_frames.size == 0:
            num_frames = trial.vicon_endpts[1] - trial.vicon_endpts[0]
            torso_pos = np.full((num_frames, 3), np.nan)
            torso_eul = np.full((num_frames, 3), np.nan)
            base_frame_inv = None
        else:
            if base_frame_inv is None:
                base_frame_inv = ht_inv(torso_truncated[present_frames[0]])
            torso_intrinsic = change_cs(base_frame_inv, torso_truncated)
            torso_pos = torso_intrinsic[:, :3, 3]
            torso_eul = np.rad2deg(zxy_intrinsic(torso_intrinsic))
        return torso_pos, torso_eul, base_frame_inv

    trial = trial_row['Trial']
    torso_pos_smoothed, torso_eul_smoothed, base_inv = process_trial(trial, 'smoothed')
    torso_pos_labeled, torso_eul_labeled, _ = process_trial(trial, 'labeled', base_inv)
    torso_pos_filled, torso_eul_filled, _ = process_trial(trial, 'filled', base_inv)

    # graph
    frame_nums = np.arange(trial.vicon_endpts[0], trial.vicon_endpts[1]) + 1
    return RawSmoothedKineTorsoPlotter(trial_row['Trial_Name'], torso_pos_labeled, torso_eul_labeled, torso_pos_filled,
                                       torso_eul_filled, torso_pos_smoothed, torso_eul_smoothed, frame_nums)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ..general.arg_parser import mod_arg_parser
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.database import create_db, anthro_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubjectV3D
    from biplane_kine.misc.json_utils import Params
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch compare labeled, filled, vs smoothed torso kinematics..',
                                     __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    anthro = anthro_db(params.biplane_vicon_db_dir)

    def armpit_thickness(subj_name):
        return anthro.loc[subj_name, 'Armpit_Thickness']

    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubjectV3D, armpit_thickness=armpit_thickness)
    db['Plotter'] = db.apply(trial_plotter, axis=1)

    # create plots
    init_graphing()
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting kinematics for subject %s', subject_name)
        subject_pdf_file = root_path / (subject_name + '.pdf')
        with PdfPages(subject_pdf_file) as subject_pdf:
            for plotter in subject_df['Plotter']:
                log.info('Outputting kinematics for trial %s', plotter.trial_name)
                figs = plotter.plot()
                subject_pdf.savefig(figs[3])
                subject_pdf.savefig(figs[2])
                for fig in figs:
                    fig.clf()
                    plt.close(fig)
