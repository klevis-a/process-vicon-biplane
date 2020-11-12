"""Export glenohumeral kinematics for three.js animations for every trial in the biplane/Vicon
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
    import shutil
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.kinematics.cs import change_cs, ht_inv
    from ..general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Export glehonhumeral kinematics for three.js', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)


    def export_to_csv(file_name, export_data):
        header_line = 'pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w'
        np.savetxt(file_name, np.concatenate(export_data, axis=1), delimiter=',', fmt='%.11g', comments='',
                   header=header_line)

    def trial_exporter(trial, subject_folder):
        gh = change_cs(ht_inv(trial.scapula_fluoro), trial.humerus_fluoro)
        pos = gh[:, :3, 3]
        quat = Rot.from_matrix(gh[:, :3, :3]).as_quat()
        trial_file = subject_folder / (trial.trial_name + '.csv')
        export_to_csv(trial_file, [pos, quat])


    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)

    # export
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting GH kinematics for subject %s', subject_name)
        subject_dir = root_path / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        for idx, t in enumerate(subject_df['Trial']):
            if idx == 0:
                shutil.copy(t.subject.humerus_landmarks_file, subject_dir)
            log.info('Outputting GH kinematics for trial %s', t.trial_name)
            trial_exporter(t, subject_dir)
