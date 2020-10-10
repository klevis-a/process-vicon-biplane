"""Time Database

This script simply creates a Vicon/biplane fluoroscope Pandas dataframe from the unerlying filesystem-based database.
The path to a config directory (containing parameters.json) must be passed in as an argument.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    from pathlib import Path
    import psutil
    import time
    from .arg_parser import mod_arg_parser
    from biplane_kine.misc.file_utils import pretty_size
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.misc.json_utils import Params

    config_dir = Path(mod_arg_parser('Time database creation', __package__, __file__))

    process = psutil.Process(os.getpid())
    print(pretty_size(process.memory_info().rss))
    params = Params.get_params(config_dir / 'parameters.json')

    start = time.time()
    db = create_db(params.biplane_vicon_db_dir, ViconCsvSubject)
    end = time.time()

    print(end - start)

    print(pretty_size(process.memory_info().rss))
