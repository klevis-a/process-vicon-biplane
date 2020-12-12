"""Graph smoothed scapula/humerus kinematics for every trial in the biplane/Vicon filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing exported biplane fluoroscopy data
output_dir: Path to where PDF file summaries should be created
num_frames_avg: Number of frames to average when using the mean smoother.
"""


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from biokinepy.euler_angles import st_isb, ht_isb
    from biokinepy.trajectory import smooth_trajectory
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject, trajectories_from_trial
    from biplane_kine.graphing.kine_plotters import RawSmoothSegmentPlotter
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.misc.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch plot scapula/humerus smoothed kinematics', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    def trial_plotter(trial, dt, num_frames_avg, torso_def):
        torso, scap_raw, hum_raw = trajectories_from_trial(trial, dt, base_cs='vicon', torso_def=torso_def)

        # smooth
        scap_smooth = smooth_trajectory(scap_raw, num_frames_avg)
        hum_smooth = smooth_trajectory(hum_raw, num_frames_avg)

        # express in the first torso frame (I don't want to express in torso because that will add noise)
        scap_raw_torso = scap_raw.in_frame(torso.ht[0])
        hum_raw_torso = hum_raw.in_frame(torso.ht[0])
        scap_smooth_torso = scap_smooth.in_frame(torso.ht[0])
        hum_smooth_torso = hum_smooth.in_frame(torso.ht[0])

        scap_plotter = \
            RawSmoothSegmentPlotter(trial.trial_name, 'Scapula', scap_raw_torso.pos,
                                    np.rad2deg(scap_raw_torso.euler.st_isb), scap_smooth_torso.pos,
                                    np.rad2deg(scap_smooth_torso.euler.st_isb), scap_smooth_torso.vel,
                                    np.rad2deg(scap_smooth_torso.ang_vel), torso.frame_nums, st_isb.legend_short)

        hum_plotter = \
            RawSmoothSegmentPlotter(trial.trial_name, 'Humerus', hum_raw_torso.pos,
                                    np.rad2deg(hum_raw_torso.euler.ht_isb), hum_smooth_torso.pos,
                                    np.rad2deg(hum_smooth_torso.euler.ht_isb), hum_smooth_torso.vel,
                                    np.rad2deg(hum_smooth_torso.ang_vel), torso.frame_nums, ht_isb.legend_short,
                                    fig_num_start=8)

        return scap_plotter, hum_plotter

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    db['Scapula_Plotter'], db['Humerus_Plotter'] = \
        zip(*db['Trial'].apply(trial_plotter, args=[db.attrs['dt'], params.num_frames_avg, params.torso_def]))

    # plot
    init_graphing()
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting scapula/humerus kinematics for subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for trial, scap_plotter, hum_plotter in zip(subject_df['Trial'], subject_df['Scapula_Plotter'],
                                                    subject_df['Humerus_Plotter']):
            log.info('Outputting scapula/humerus kinematics for trial %s', trial.trial_name)
            scap_hum_figs = (scap_plotter.plot(), hum_plotter.plot())
            trial_pdf_file = subject_dir / (trial.trial_name + '.pdf')
            with PdfPages(trial_pdf_file) as trial_pdf:
                for segment_figs in scap_hum_figs:
                    for idx, fig in enumerate(segment_figs):
                        if idx in (0, 1, 5, 6, 7):
                            trial_pdf.savefig(fig)
                        fig.clf()
                        plt.close(fig)
