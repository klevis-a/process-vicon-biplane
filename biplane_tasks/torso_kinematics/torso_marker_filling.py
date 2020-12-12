"""Graph the filled torso marker cluster for every trial in the biplane/Vicon filesystem-based database

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
    from biplane_kine.graphing.kine_plotters import MarkerClusterFillPlotter
    from biokinepy.filling import fill_gaps_rb
    from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception
    from biplane_kine.misc.arg_parser import mod_arg_parser
    from ..parameters import read_filling_directives, trial_filling_directives
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch plot filled torso marker cluster', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    def trial_plotter(trial, marker_nms, filling_direct_all):
        log.info('Processing trial %s', trial.trial_name)
        markers_to_fill = trial_filling_directives(filling_direct_all, trial.trial_name)
        gaps_filled = {}
        filled_data = {}
        sfs_data = {}

        if not markers_to_fill:
            return None

        for (marker_name, fill_from) in markers_to_fill.items():
            assert(marker_name not in fill_from)
            filled_marker, gaps = fill_gaps_rb(trial.smoothed[marker_name],
                                               np.stack([trial.smoothed[fill_source] for fill_source in fill_from], 0))
            gaps_filled[marker_name] = gaps
            _, _, _, smoothed = piecewise_filter_with_exception({}, filled_marker, filled_marker, db.attrs['dt'],
                                                                white_noise_var=100000)
            filled_data[marker_name] = filled_marker
            sfs_data[marker_name] = np.full_like(filled_marker, np.nan)
            sfs_data[marker_name][smoothed.endpts[0]:smoothed.endpts[1], :] = smoothed.means.pos

        smooth_marker_data = np.stack([trial.smoothed[marker_name] for marker_name in marker_nms], 0)
        return MarkerClusterFillPlotter(trial.trial_name, smooth_marker_data, marker_names, gaps_filled,
                                        markers_to_fill, filled_data, sfs_data, trial.vicon_endpts)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    marker_names = ['T10', 'T5', 'STRN', 'C7', 'CLAV']
    filling_directives_all = read_filling_directives(params.filling_directives)
    db['Plotter'] = db['Trial'].apply(trial_plotter, args=[marker_names, filling_directives_all])

    # create plots
    init_graphing()
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting marker availability for subject %s', subject_name)
        subject_pdf_file = root_path / (subject_name + '.pdf')
        with PdfPages(subject_pdf_file) as subject_pdf:
            for plotter in subject_df['Plotter']:
                if plotter is not None:
                    log.info('Outputting marker availability for trial %s', plotter.trial_name)
                    figs = plotter.plot()
                    for fig in figs:
                        subject_pdf.savefig(fig)
                        fig.clf()
                        plt.close(fig)
