"""Compute differences between V3D and ISB torso coordinate sytems in the static trial for every subject

This script iterates over the Vicon/biplane fluoroscopy filesystem-based database and computes the torso CS using the
method in Visual3D and the method specified by ISB. It then computes the position and orientation difference between
the two CS specifications for every subject.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import pandas as pd
    import numpy as np
    import quaternion
    from ..general.arg_parser import mod_arg_parser
    from biplane_kine.database import create_db, anthro_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.kinematics.joint_cs import torso_cs_isb, torso_cs_v3d
    from biplane_kine.misc.json_utils import Params
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Compare torso CS as defined via Visual3D and ISB', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    anthro = anthro_db(params.biplane_vicon_db_dir)

    # create a dataframe of just the subjects
    subjects = pd.Series(db['Subject'].unique())
    subject_df = pd.DataFrame({'Subject_Name': subjects.apply(lambda subj: subj.subject_name), 'Subject': subjects})

    def torso_isb(subj):
        return torso_cs_isb(subj.static['C7'], subj.static['CLAV'], subj.static['STRN'], subj.static['T10'],
                            subj.static['T5'])

    def torso_v3d(subj, anthro):
        return torso_cs_v3d(subj.static['C7'], subj.static['CLAV'], subj.static['STRN'], subj.static['T10'],
                            subj.static['RGTR'], subj.static['LGTR'], subj.static['LSHO'], subj.static['RSH0'],
                            anthro.loc[subj.subject_name, 'Armpit_Thickness'])

    # compute ISB and torso CS
    subject_df['isb'] = subject_df['Subject'].apply(torso_isb)
    subject_df['v3d'] = subject_df['Subject'].apply(torso_v3d, args=[anthro])

    def frame_diff(df_row):
        pos_diff = df_row['v3d'][0:3, 3] - df_row['isb'][0:3, 3]
        matrix_diff = df_row['isb'][0:3, 0:3].T @ df_row['v3d'][0:3, 0:3]
        pose_diff = np.full((6,), np.nan)
        pose_diff[0:3] = pos_diff
        pose_diff[3:] = np.rad2deg(quaternion.as_rotation_vector(quaternion.from_rotation_matrix(matrix_diff,
                                                                                                 nonorthogonal=False)))
        return pose_diff

    (subject_df['pos_x'], subject_df['pos_y'], subject_df['pos_z'], subject_df['rot_x'], subject_df['rot_y'],
     subject_df['rot_z']) = zip(*(subject_df[['v3d', 'isb']].apply(frame_diff, axis=1)))
