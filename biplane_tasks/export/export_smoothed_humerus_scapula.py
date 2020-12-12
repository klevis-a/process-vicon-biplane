"""Export smoothed scapula and humerus kinematics for for every trial in the biplane/Vicon filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where trials should be exported
num_frames_avg: Number of frames to average when smoothing scapula/humerus trajectories.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import numpy as np
    from biplane_kine.database import create_db
    from biokinepy.mean_smoother import smooth_quat_traj, smooth_pos_traj
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.misc.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Export smoothed scapula and humerus kinematics',
                                     __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)


    def export_to_csv(file_name, frame, pos_data, quat_data):
        header_line = ['frame', 'pos_x', 'pos_y', 'pos_z', 'quat_w', 'quat_x', 'quat_y', 'quat_z']
        header_line_str = ','.join(header_line)
        np.savetxt(file_name, np.concatenate((frame[..., np.newaxis], pos_data, quat_data), axis=1), delimiter=',',
                   fmt='%.11g', comments='', header=header_line_str)

    def trial_exporter(trial, subject_folder, num_frames_avg):
        scap_quat = trial.scapula_quat_fluoro
        scap_pos = trial.scapula_pos_fluoro
        hum_quat = trial.humerus_quat_fluoro
        hum_pos = trial.humerus_pos_fluoro
        scap_quat_smooth = smooth_quat_traj(scap_quat, num_frames_avg)
        scap_pos_smooth = smooth_pos_traj(scap_pos, num_frames_avg)
        hum_quat_smooth = smooth_quat_traj(hum_quat, num_frames_avg)
        hum_pos_smooth = smooth_pos_traj(hum_pos, num_frames_avg)

        trial_dir = subject_folder / trial.trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        export_to_csv(trial_dir / (trial.trial_name + '_humerus_biplane_avgSmooth.csv'),
                      trial.humerus_frame_nums, hum_pos_smooth, hum_quat_smooth)
        export_to_csv(trial_dir / (trial.trial_name + '_scapula_biplane_avgSmooth.csv'),
                      trial.scapula_frame_nums, scap_pos_smooth, scap_quat_smooth)


    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)

    # export
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting scapula, humerus kinematics for subject %s', subject_name)
        subject_dir = root_path / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        for t in subject_df['Trial']:
            log.info('Outputting scapula, humerus kinematics for trial %s', t.trial_name)
            trial_exporter(t, subject_dir, params.num_frames_avg)
