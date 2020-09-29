from biplane_kine.database.c3d_helper import C3DHelper
from biplane_kine.kinematics.cs import vec_transform
from biplane_kine.graphing.plotters import AccuracySmoothingPlotter
from biplane_kine.graphing.graph_utils import init_graphing
import matplotlib.pyplot as plt


def add_c3d_helper(db_df, vicon_labeled_path):
    def create_c3d_helper(row, base_path):
        return C3DHelper(str(Path(base_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')))

    db_df['C3D_Trial'] = db_df.apply(create_c3d_helper, axis=1, args=[vicon_labeled_path])


def compute_accuracy_diff(row):
    acc_trial = row['Accuracy_Trial']
    c3d_trial = row['C3D_Trial']

    diff_by_marker = {}
    for marker, acc_marker_data in acc_trial:
        # vicon marker data accounting for endpoints (to match biplane data)
        vicon_marker_data = c3d_trial[marker][acc_trial.vicon_endpts[0]:acc_trial.vicon_endpts[1]]
        # select just the frames that were tracked in the accuracy trial
        vmd_trunc = vicon_marker_data[acc_marker_data.indices]
        # transform vicon marker data to fluoro CS
        vmd_fluoro = vec_transform(acc_trial.subject.f_t_v, vmd_trunc)
        # compute difference and store
        diff_acc_vicon = vmd_fluoro[:, :3] - acc_marker_data.data
        diff_by_marker[marker] = diff_acc_vicon

    return diff_by_marker


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    from biplane_kine.database import create_db
    from biplane_kine.database.vicon_accuracy import ViconAccuracySubjectEndpts
    from biplane_kine.misc.json_utils import Params
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    db = create_db(params.accuracy_db_dir, ViconAccuracySubjectEndpts)
    add_c3d_helper(db, params.labeled_c3d_dir)
    db['diff'] = db.apply(compute_accuracy_diff, axis=1)
    trial_row = db.loc['N002A_CA_t01']
    init_graphing()
    plotter = AccuracySmoothingPlotter('N002A_CA_t01', 'RACRM', trial_row['diff']['RACRM'],
                                       trial_row['Accuracy_Trial']['RACRM'].frames)
    plotter.plot()
    plt.show()
