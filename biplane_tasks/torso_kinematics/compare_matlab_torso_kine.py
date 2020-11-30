"""Graph differences between computed torso frame trajectory from Matlab computations

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
matlab_threejs_dir: Path to directory for matlab has exported data for three.js
trial_name: the name of the trial to perform comparisons for
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import quaternion
    from biplane_kine.misc.arg_parser import mod_arg_parser
    from biplane_kine.graphing.common_graph_utils import init_graphing, make_interactive
    from biplane_kine.kinematics.vec_ops import extended_dot
    from biplane_kine.kinematics.cs import change_cs
    from biplane_kine.kinematics.segments import StaticTorsoSegment
    from biplane_kine.kinematics.absor import compute_trajectory
    from biplane_kine.database import create_db, anthro_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubjectV3D
    from biplane_kine.misc.json_utils import Params
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Compare Python vs Matlab torso kinematics', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    anthro = anthro_db(params.biplane_vicon_db_dir)

    def armpit_thickness(subject_name):
        return anthro.loc[subject_name, 'Armpit_Thickness']

    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubjectV3D, armpit_thickness=armpit_thickness)

    # retrieve trial information
    trial_row = db.loc[params.trial_name]
    subject_id = trial_row['Subject_Name']
    trial = trial_row['Trial']
    tracking_markers = np.stack([trial.filled[marker] for marker in StaticTorsoSegment.TRACKING_MARKERS], 0)
    torso_vicon = compute_trajectory(trial.subject.torso.static_markers_intrinsic, tracking_markers)
    torso_fluoro = change_cs(trial.subject.f_t_v, torso_vicon)
    torso_truncated = torso_fluoro[trial.vicon_endpts[0]:trial.vicon_endpts[1]]
    quat = quaternion.as_float_array(quaternion.from_rotation_matrix(torso_truncated[:, :3, :3], nonorthogonal=False))
    torso_pos_quat = np.concatenate((torso_truncated[:, :3, 3], quat), 1)

    # retrieve reference data
    ref_data = pd.read_csv(Path(params.matlab_threejs_dir) / subject_id / (params.trial_name + '.csv'))

    # since the negative of a quaternions represents the same orientation make the appropriate sign flip so graphs align
    mask = extended_dot(torso_pos_quat[:, 3:], ref_data.iloc[:, 3:7].to_numpy()) < 0
    torso_pos_quat[mask, 3:] = -torso_pos_quat[mask, 3:]

    # graph
    init_graphing()
    for n in range(7):
        plt.figure(n+1)
        plt.plot(torso_pos_quat[:, n], 'r')
        plt.plot(ref_data.iloc[:, n], 'g')
        make_interactive()
    plt.show()
