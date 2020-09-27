from biplane_kine.database.c3d_helper import C3DHelper


def add_c3d_helper(db_df, vicon_labeled_path):
    def create_c3d_helper(row, base_path):
        return C3DHelper(str(Path(base_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')))

    db_df['C3D_Trial'] = db_df.apply(create_c3d_helper, axis=1, args=[vicon_labeled_path])


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
