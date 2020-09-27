if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    from pathlib import Path
    import psutil
    import time
    import sys
    from biplane_kine.misc.file_utils import pretty_size
    from biplane_kine.database import create_db
    from biplane_kine.database.dynamic_subject import DynamicSubject
    from biplane_kine.misc.json_utils import Params

    process = psutil.Process(os.getpid())
    print(pretty_size(process.memory_info().rss))

    start = time.time()
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')
    db = create_db(params.db_dir, DynamicSubject)
    end = time.time()
    print(end - start)

    print(pretty_size(process.memory_info().rss))
