"""Export torso, scapula, and humerus kinematics for three.js animations for every trial in the biplane/Vicon
filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where trials should be exported
torso_def: Whether to use the ISB or V3D definition for establishing the torso coordinate system.
"""

import numpy as np
from biplane_kine.database.biplane_vicon_db import trajectories_from_trial


def export_to_csv(file_name, torso_data, scapula_data, hum_data):
    header_line_generic = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    header_line_torso = ['torso_' + col for col in header_line_generic]
    header_line_scapula = ['scap_' + col for col in header_line_generic]
    header_line_humerus = ['hum_' + col for col in header_line_generic]
    header_line = ','.join([','.join(header_line_torso), ','.join(header_line_scapula),
                            ','.join(header_line_humerus)])
    np.savetxt(file_name, np.concatenate((torso_data, scapula_data, hum_data), axis=1), delimiter=',', fmt='%.11g',
               comments='', header=header_line)


def trial_exporter(trial, dt, subject_folder, torso_def):
    def pos_quat_from_traj(traj):
        quat_sf = traj.quat_float
        quat_sl = np.concatenate((quat_sf[:, 1:], quat_sf[:, 0][..., np.newaxis]), 1)
        return np.concatenate((traj.pos, quat_sl), axis=1)

    torso, scap, hum = trajectories_from_trial(trial, dt, smoothed=True, torso_def=torso_def)
    torso_pos_quat = pos_quat_from_traj(torso)
    scapula_pos_quat = pos_quat_from_traj(scap)
    humerus_pos_quat = pos_quat_from_traj(hum)

    trial_file = subject_folder / (trial.trial_name + '.csv')
    export_to_csv(trial_file, torso_pos_quat, scapula_pos_quat, humerus_pos_quat)
    return trial_file


def no_static_dir(file):
    return file.parts[-3] + '/' + file.parts[-1]


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import json
    import shutil
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.misc.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    friendly_names = {'SA': 'Scapular Plane Abduction',
                      'CA': 'Coronal Plane Abduction',
                      'FE': 'Forward Elevation',
                      'ERa90': 'External Rotation at 90&deg; of Abduction',
                      'ERaR': 'External Rotation at Rest',
                      'WCA': 'Weighted Coronal Plane Abduction',
                      'WSA': 'Weighted Scapular Plane Abduction',
                      'WFE': 'Weighted Forward Elevation'}

    friendly_names_keys = list(friendly_names.keys())

    sorterIndex = dict(zip(friendly_names_keys, range(len(friendly_names_keys))))

    # initialize
    config_dir = Path(mod_arg_parser('Export torso, scapula, and humerus kinematics for three.js',
                                     __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    json_export = {}

    # export
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting torso, scapula, humerus kinematics for subject %s', subject_name)
        subject_dir = root_path / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        json_export[subject_name] = {}
        json_export[subject_name]['config'] = {}
        json_export[subject_name]['activities'] = {}

        # sort by the friendly_names order above
        subject_df = subject_df.copy()
        subject_df['activity_rank'] = subject_df['Activity'].map(sorterIndex)
        subject_df.sort_values('activity_rank', ascending=True, inplace=True)
        for idx, (t, activity) in enumerate(zip(subject_df['Trial'], subject_df['Activity'])):
            if idx == 0:
                shutil.copy(t.subject.humerus_landmarks_file, subject_dir)
                shutil.copy(t.subject.scapula_landmarks_file, subject_dir)
                shutil.copy(t.subject.humerus_stl_smooth_file, subject_dir)
                shutil.copy(t.subject.scapula_stl_smooth_file, subject_dir)
                json_export[subject_name]['config']['humerus_landmarks_file'] = \
                    no_static_dir(t.subject.humerus_landmarks_file)
                json_export[subject_name]['config']['scapula_landmarks_file'] = \
                    no_static_dir(t.subject.scapula_landmarks_file)
                json_export[subject_name]['config']['humerus_stl_file'] = \
                    no_static_dir(t.subject.humerus_stl_smooth_file)
                json_export[subject_name]['config']['scapula_stl_file'] = \
                    no_static_dir(t.subject.scapula_stl_smooth_file)
            if activity in (['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'WCA', 'WSA', 'WFE']):
                log.info('Outputting torso, scapula, humerus kinematics for trial %s', t.trial_name)
                f = trial_exporter(t, db.attrs['dt'], subject_dir, params.torso_def)
                json_export[subject_name]['activities'][friendly_names[activity]] = {}
                json_export[subject_name]['activities'][friendly_names[activity]]['trajectory'] = (f.parts[-2] + '/' +
                                                                                                   f.parts[-1])
                json_export[subject_name]['activities'][friendly_names[activity]]['freq'] = round(1/db.attrs['dt'], 2)

    with open(root_path / 'db_summary.json', 'w') as summary_file:
        json.dump(json_export, summary_file, indent=4)
