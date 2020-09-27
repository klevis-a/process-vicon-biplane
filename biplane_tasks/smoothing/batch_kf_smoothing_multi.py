if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    import multiprocessing as mp
    from functools import partial
    from biplane_kine.database import create_db
    from biplane_kine.database.dynamic_subject import DynamicSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.graph_utils import init_graphing
    from ..parameters import read_smoothing_exceptions
    from .batch_kf_smoothing import trial_plotter

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.db_dir, DynamicSubject)
    init_graphing()
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)

    # create plots
    for subject_name, subject_df in db.groupby('Subject'):
        print('Smoothing subject {}'.format(subject_name))
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_trial_plotter = partial(trial_plotter, dt=db.attrs['dt'], subj_dir=subject_dir,
                                        all_smoothing_except=all_exceptions)
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(subject_trial_plotter, subject_df['Trial'])
