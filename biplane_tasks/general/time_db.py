if __name__ == '__main__':
    import os
    from pathlib import Path
    import psutil
    import time
    import sys
    from biplane_kine.misc.file_utils import pretty_size
    from biplane_kine.database import create_db
    from biplane_kine.misc.json_utils import Params

    process = psutil.Process(os.getpid())
    print(pretty_size(process.memory_info().rss))

    start = time.time()
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')
    db_dir = params.db_dir
    db, anthro = create_db(db_dir)
    end = time.time()
    print(end - start)

    print(pretty_size(process.memory_info().rss))
