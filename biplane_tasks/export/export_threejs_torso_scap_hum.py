"""Export torso, scapula, and humerus kinematics for three.js animations for every trial in the biplane/Vicon
filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where trials should be exported
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import json
    import shutil
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.kinematics.cs import change_cs, ht_inv
    from biplane_tasks.general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Export torso, scapula, and humerus kinematics for three.js',
                                     __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)


    def export_to_csv(file_name, torso_data, scapula_data, hum_data):
        header_line_generic = ['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
        header_line_torso = ['torso_' + col for col in header_line_generic]
        header_line_scapula = ['scap_' + col for col in header_line_generic]
        header_line_humerus = ['hum_' + col for col in header_line_generic]
        header_line = ','.join([','.join(header_line_torso), ','.join(header_line_scapula),
                                ','.join(header_line_humerus)])
        np.savetxt(file_name, np.concatenate((torso_data, scapula_data, hum_data), axis=1), delimiter=',', fmt='%.11g',
                   comments='', header=header_line)

    def trial_exporter(trial, subject_folder, torso_source):
        torso_vicon = getattr(trial, torso_source)[trial.vicon_endpts[0]:trial.vicon_endpts[1]]
        torso_vicon = torso_vicon[trial.humerus_biplane_data.index.to_numpy() - 1]
        scapula_vicon = change_cs(ht_inv(trial.subject.f_t_v), trial.scapula_fluoro)
        humerus_vicon = change_cs(ht_inv(trial.subject.f_t_v), trial.humerus_fluoro)

        def pos_quat_from_ht(mat):
            pos = mat[:, :3, 3]
            quat = Rot.from_matrix(mat[:, :3, :3]).as_quat()
            return np.concatenate((pos, quat), axis=1)

        torso_pos_quat = pos_quat_from_ht(torso_vicon)
        scapula_pos_quat = pos_quat_from_ht(scapula_vicon)
        humerus_pos_quat = pos_quat_from_ht(humerus_vicon)

        trial_file = subject_folder / (trial.trial_name + '.csv')
        export_to_csv(trial_file, torso_pos_quat, scapula_pos_quat, humerus_pos_quat)
        return trial_file


    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    json_export = {}

    def no_static_dir(file):
        return file.parts[-3] + '/' + file.parts[-1]

    # export
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting torso, scapula, humerus kinematics for subject %s', subject_name)
        subject_dir = root_path / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        json_export[subject_name] = {}
        json_export[subject_name]['config'] = {}
        json_export[subject_name]['activities'] = {}
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
                f = trial_exporter(t, subject_dir, params.threejs_torso_source)
                json_export[subject_name]['activities'][activity] = f.parts[-2] + '/' + f.parts[-1]

    with open(root_path / 'db_summary.json', 'w') as summary_file:
        json.dump(json_export, summary_file, indent=4)