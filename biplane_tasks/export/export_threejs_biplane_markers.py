"""Export torso, scapula, and humerus kinematics from biplane fluoroscopy and Vicon marker trajectories for three.js
animations for every trial in the biplane/Vicon filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where trials should be exported
torso_def: Whether to use the ISB or V3D definition for establishing the torso coordinate system.
"""

import numpy as np


def marker_exporter(trial, subject_folder):
    col_names = trial.vicon_csv_data_smoothed.columns.values.tolist()
    col_names_export = [col_name.replace('.1', '_Y').replace('.2', '_Z')
                        if '.' in col_name else (col_name + '_X') for col_name in col_names]
    file_path = subject_folder / (trial.trial_name + '_markers.csv')
    marker_data = trial.vicon_csv_data_smoothed.to_numpy()[trial.vicon_endpts[0]:
                                                           trial.vicon_endpts[1]][trial.humerus_frame_nums - 1]
    np.savetxt(file_path, marker_data, delimiter=',', fmt='%.11g', comments='', header=','.join(col_names_export))
    return file_path


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
    from biplane_tasks.export.export_threejs_torso_scap_hum import no_static_dir, trial_exporter
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Export torso, scapula, and humerus kinematics and Vicon marker trajectories '
                                     'for three.js',
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
        json_export[subject_name]['markers'] = {}
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
                json_export[subject_name]['config']['humerus_stl_smooth_file'] = \
                    no_static_dir(t.subject.humerus_stl_smooth_file)
                json_export[subject_name]['config']['scapula_stl_smooth_file'] = \
                    no_static_dir(t.subject.scapula_stl_smooth_file)
            if activity in (['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'WCA', 'WSA', 'WFE']):
                log.info('Outputting torso, scapula, humerus kinematics for trial %s', t.trial_name)
                f = trial_exporter(t, db.attrs['dt'], subject_dir, params.torso_def)
                json_export[subject_name]['activities'][activity] = f.parts[-2] + '/' + f.parts[-1]

                log.info('Outputting marker trajectories for trial %s', t.trial_name)
                markers_file = marker_exporter(t, subject_dir)
                json_export[subject_name]['markers'][activity] = markers_file.parts[-2] + '/' + markers_file.parts[-1]

    with open(root_path / 'db_summary.json', 'w') as summary_file:
        json.dump(json_export, summary_file, indent=4)
