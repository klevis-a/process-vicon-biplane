import matplotlib.pyplot as plt
import numpy as np
import distutils.util
from biplane_kine.database.c3d_helper import C3DTrial
from biplane_kine.kinematics.cs import vec_transform
from biplane_tasks.parameters import marker_smoothing_exceptions
from biplane_kine.graphing.vicon_accuracy_plotters import ViconAccuracySmoothingPlotter
from biplane_kine.graphing.common_graph_utils import init_graphing
from biplane_kine.smoothing.kf_filtering_helpers import kf_filter_marker_piecewise, combine_pieces


def add_c3d_helper(db_df, vicon_labeled_path, vicon_filled_path):
    def create_c3d_helper(row, labeled_path, filled_path):
        return C3DTrial(str(Path(labeled_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')),
                        str(Path(filled_path) / row['Subject_Name'] / (row['Trial_Name'] + '.c3d')))

    db_df['C3D_Trial'] = db_df.apply(create_c3d_helper, axis=1, args=[vicon_labeled_path, vicon_filled_path])


def compute_accuracy_diff(row, all_except, dt):
    biplane_trial = row['Biplane_Marker_Trial']
    c3d_trial = row['C3D_Trial']
    endpts = biplane_trial.vicon_endpts

    plotter_by_marker = {}
    for marker, acc_marker_data in biplane_trial:
        marker_exceptions = marker_smoothing_exceptions(all_except, row['Trial_Name'], marker)
        should_use = bool(distutils.util.strtobool(marker_exceptions.get('use_marker', 'True')))
        if not should_use:
            log.warning('Skipping marker because it is labeled as DO NOT USE.')
            plotter_by_marker[marker] = None
            continue
        smoothing_params = marker_exceptions.get('smoothing_params', {})
        frame_ignores = np.asarray(marker_exceptions.get('frame_ignores', []))

        # before ignoring frames make a copy of the data so we have it for computation purposes
        vicon_marker_data = c3d_trial.labeled[marker][endpts[0]:endpts[1]].copy()

        # ignore frames
        if frame_ignores.size > 0:
            c3d_trial.labeled[marker][frame_ignores - 1, :] = np.nan

        # filter
        _, smoothed_pieces = kf_filter_marker_piecewise(c3d_trial, marker, dt, **smoothing_params)
        smoothed = combine_pieces(smoothed_pieces)

        # now make sure that the smoothed data extends from endpts[0] to endpts[1]
        smoothed_rectified = np.full((endpts[1]-endpts[0], 3), np.nan)
        source_start_idx = endpts[0]-smoothed.endpts[0] if smoothed.endpts[0] < endpts[0] else 0
        source_stop_idx = endpts[1] - smoothed.endpts[0] if smoothed.endpts[1] > endpts[1] \
            else smoothed.endpts[1] - smoothed.endpts[0]
        target_start_idx = smoothed.endpts[0] - endpts[0] if smoothed.endpts[0] > endpts[0] else 0
        target_stop_idx = target_start_idx + (source_stop_idx - source_start_idx)
        smoothed_rectified[target_start_idx:target_stop_idx, :] = \
            smoothed.means.pos[source_start_idx:source_stop_idx, :]

        # transform vicon marker data to fluoro CS
        vmd_fluoro = vec_transform(biplane_trial.subject.f_t_v, vicon_marker_data)[:, :3]
        smoothed_vmd_fluoro = vec_transform(biplane_trial.subject.f_t_v,
                                            np.concatenate((smoothed_rectified, np.ones((smoothed_rectified.shape[0], 1))), axis=1))
        # create plotter
        plotter_by_marker[marker] = ViconAccuracySmoothingPlotter(row['Trial_Name'], marker, acc_marker_data,
                                                                  vmd_fluoro, smoothed_vmd_fluoro[:, :3])
    return plotter_by_marker


def trial_plotter(row, subj_dir):
    trial_name = row['Trial_Name']
    log.info('Outputting trial %s', trial_name)
    trial_dir = subj_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_pdf_file = subject_dir / (trial_name + '.pdf')
    plotter_by_marker = row['Vicon_Accuracy_Plotter']
    with PdfPages(trial_pdf_file) as trial_pdf:
        for (marker, plotter) in plotter_by_marker.items():
            figs = plotter.plot()
            for (fig_num, fig) in enumerate(figs):
                trial_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    from biplane_tasks.parameters import read_smoothing_exceptions
    from biplane_kine.database import create_db
    from biplane_kine.database.vicon_accuracy import BiplaneMarkerSubjectEndpts
    from biplane_kine.misc.json_utils import Params
    from matplotlib.backends.backend_pdf import PdfPages
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
    db = create_db(params.accuracy_db_dir, BiplaneMarkerSubjectEndpts)
    add_c3d_helper(db, params.labeled_c3d_dir, params.filled_c3d_dir)
    db['Vicon_Accuracy_Plotter'] = db.apply(compute_accuracy_diff, axis=1, args=[all_exceptions, db.attrs['dt']])

    root_path = Path(params.output_dir)
    init_graphing()
    # plot markers
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        subject_df.apply(trial_plotter, axis=1, args=[subject_dir])
