"""Batch smooth Vicon marker data using Kalman smoothing (multi-threaded)

This script iterates over the Vicon/biplane fluoroscpy filesystem-based database and creates PDF records of the effects
of smoothing. Uses the multiprocessing module, although logging (only tested on Windows) doesn't work.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to the directory where PDF records for each marker (and trial) will be output.
smoothing_exceptions: Path to a file containing smoothing exceptions for each trial/marker.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import multiprocessing as mp
    from functools import partial
    from ..general.arg_parser import mod_arg_parser
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from ..parameters import read_smoothing_exceptions
    from .batch_kf_smoothing import trial_plotter

    # initialize
    config_dir = Path(mod_arg_parser('Time database creation', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, ViconCsvSubject)
    init_graphing()
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)

    # create plots
    for subject_name, subject_df in db.groupby('Subject_Name'):
        print('Smoothing subject {}'.format(subject_name))
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_trial_plotter = partial(trial_plotter, dt=db.attrs['dt'], subj_dir=subject_dir,
                                        all_smoothing_except=all_exceptions)
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(subject_trial_plotter, subject_df['Trial'])
