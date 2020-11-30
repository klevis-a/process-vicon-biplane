"""Export torso ISB-defined kinematics as derived from smoothed/filled/smoothed marker positions for every trial in the
biplane/Vicon filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where to export kinematics
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import numpy as np
    import quaternion
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.smoothing.filling import fill_gaps_rb
    from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception
    from biplane_kine.kinematics.segments import StaticTorsoSegment
    from biplane_kine.kinematics.absor import compute_trajectory_continuous
    from biplane_kine.misc.arg_parser import mod_arg_parser
    from biplane_tasks.parameters import read_filling_directives, trial_filling_directives
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Export ISB torso kinematics', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    def trial_exporter(trial, marker_names, filling_directives_all, dt, subject_folder):
        log.info('Exporting trial %s', trial.trial_name)
        markers_to_fill = trial_filling_directives(filling_directives_all, trial.trial_name)

        smooth_marker_pos = np.stack([trial.smoothed[marker_name] for marker_name in marker_names], 0)
        sfs_marker_pos = smooth_marker_pos.copy()

        # fill then smooth again
        for (marker_name, fill_from) in markers_to_fill.items():
            assert(marker_name not in fill_from)
            marker_idx = marker_names.index(marker_name)
            filled_marker, _ = fill_gaps_rb(trial.smoothed[marker_name],
                                            np.stack([trial.smoothed[fill_source] for fill_source in fill_from], 0))

            # smooth
            _, _, _, smoothed = piecewise_filter_with_exception({}, filled_marker, filled_marker, dt,
                                                                white_noise_var=100000)
            sfs_data = np.full_like(filled_marker, np.nan)
            sfs_data[smoothed.endpts[0]:smoothed.endpts[1], :] = smoothed.means.pos
            sfs_marker_pos[marker_idx] = sfs_data

        def process_trial(static_markers, tracking_markers):
            torso_traj = compute_trajectory_continuous(static_markers, tracking_markers)
            torso_pos = torso_traj[:, :3, 3]
            torso_quat = quaternion.as_float_array(quaternion.from_rotation_matrix(torso_traj[:, :3, :3],
                                                                                   nonorthogonal=False))
            return torso_pos, torso_quat

        stat_markers = trial.subject.torso.static_markers_intrinsic
        torso_pos_sfs, torso_quat_sfs = process_trial(stat_markers, sfs_marker_pos)

        def export_to_csv(file_name, export_data):
            header_line = 'pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z'
            np.savetxt(file_name, np.concatenate(export_data, axis=1), delimiter=',', fmt='%.11g', comments='',
                       header=header_line)

        trial_folder = subject_folder / trial.trial_name
        trial_folder.mkdir(parents=True, exist_ok=True)
        trial_file = trial_folder / (trial.trial_name + '_torso.csv')
        export_to_csv(trial_file, [torso_pos_sfs, torso_quat_sfs])

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)

    # export kinematics
    filling_directions = read_filling_directives(params.filling_directives)
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for t in subject_df['Trial']:
            trial_exporter(t, StaticTorsoSegment.TRACKING_MARKERS, filling_directions, db.attrs['dt'], subject_dir)
