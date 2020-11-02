"""Graph the availability of each invidual marker in the torso marker cluster for every trial in the biplane/Vicon
filesystem-based database

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing Vicon skin marker data.
output_dir: Path to where PDF file summaries should be created
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.graphing.kine_plotters import MarkerClusterAvailPlotter
    from ..general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch plot availability of torso marker cluster '
                                     'record', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    def trial_plotter(trial, marker_names):
        marker_data = np.stack([trial.smoothed[marker_name] for marker_name in marker_names], 0)
        return MarkerClusterAvailPlotter(marker_data, marker_names, trial.vicon_endpts, trial.trial_name)

    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    marker_names = ['T10', 'T5', 'STRN', 'C7', 'CLAV']
    db['Plotter'] = db['Trial'].apply(trial_plotter, args=[marker_names])

    # create plots
    init_graphing()
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting marker availability for subject %s', subject_name)
        subject_pdf_file = root_path / (subject_name + '.pdf')
        with PdfPages(subject_pdf_file) as subject_pdf:
            for plotter in subject_df['Plotter']:
                log.info('Outputting marker availability for trial %s', plotter.trial_name)
                figs = plotter.plot()
                subject_pdf.savefig(figs[0])
                figs[0].clf()
                plt.close(figs[0])
