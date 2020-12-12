"""Graph torso kinematics as derived from previously filled (in Vicon), smoothed, smoothed/filled, and
smoothed/filled/smoothed marker positions for every trial in the biplane/Vicon filesystem-based database.

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
    from biokinepy.filling import fill_gaps_rb
    from biplane_kine.smoothing.kf_filtering_helpers import piecewise_filter_with_exception
    from biplane_kine.kinematics.segments import StaticTorsoSegment
    from biokinepy.cs import ht_inv, change_cs
    from biokinepy.euler_angles import zxy_intrinsic
    from biokinepy.absor import compute_trajectory, compute_trajectory_continuous
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.graphing.kine_plotters import TorsoTrajComparisonPlotter
    from biplane_kine.misc.arg_parser import mod_arg_parser
    from ..parameters import read_filling_directives, trial_filling_directives
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch plot torso kinematics', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    def trial_plotter(trial, marker_names, filling_directives_all, dt):
        log.info('Processing trial %s', trial.trial_name)
        markers_to_fill = trial_filling_directives(filling_directives_all, trial.trial_name)

        smooth_marker_pos = np.stack([trial.smoothed[marker_name] for marker_name in marker_names], 0)
        prev_filled_marker_pos = np.stack([trial.filled[marker_name] for marker_name in marker_names], 0)
        filled_marker_pos = smooth_marker_pos.copy()
        sfs_marker_pos = smooth_marker_pos.copy()

        # fill then smooth again
        for (marker_name, fill_from) in markers_to_fill.items():
            assert(marker_name not in fill_from)
            marker_idx = marker_names.index(marker_name)
            filled_marker, _ = fill_gaps_rb(trial.smoothed[marker_name],
                                            np.stack([trial.smoothed[fill_source] for fill_source in fill_from], 0))
            filled_marker_pos[marker_idx] = filled_marker

            # smooth
            _, _, _, smoothed = piecewise_filter_with_exception({}, filled_marker, filled_marker, dt,
                                                                white_noise_var=100000)
            sfs_data = np.full_like(filled_marker, np.nan)
            sfs_data[smoothed.endpts[0]:smoothed.endpts[1], :] = smoothed.means.pos
            sfs_marker_pos[marker_idx] = sfs_data

        def process_trial(static_markers, tracking_markers, traj_func, base_frame_inv=None):
            torso_traj = traj_func(static_markers, tracking_markers)
            present_frames = np.nonzero(~np.any(np.isnan(torso_traj), (-2, -1)))[0]
            if present_frames.size == 0:
                num_frames = tracking_markers.shape[1]
                torso_pos = np.full((num_frames, 3), np.nan)
                torso_eul = np.full((num_frames, 3), np.nan)
                base_frame_inv = None
            else:
                if base_frame_inv is None:
                    base_frame_inv = ht_inv(torso_traj[present_frames[0]])
                torso_intrinsic = change_cs(base_frame_inv, torso_traj)
                torso_pos = torso_intrinsic[:, :3, 3]
                torso_eul = np.rad2deg(zxy_intrinsic(torso_intrinsic))
            return (torso_pos, torso_eul), base_frame_inv

        stat_markers = trial.subject.torso.static_markers_intrinsic
        torso_kine_smooth, base_inv = process_trial(stat_markers, smooth_marker_pos, compute_trajectory)
        torso_kine_sfs, _ = process_trial(stat_markers, sfs_marker_pos, compute_trajectory_continuous, base_inv)
        torso_kine_filled, _ = process_trial(stat_markers, filled_marker_pos, compute_trajectory_continuous, base_inv)
        torso_kine_prev_filled, _ = process_trial(stat_markers, prev_filled_marker_pos, compute_trajectory, base_inv)

        return TorsoTrajComparisonPlotter(trial.trial_name, torso_kine_prev_filled, torso_kine_smooth,
                                          torso_kine_filled, torso_kine_sfs, trial.vicon_endpts)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    db['Plotter'] = db['Trial'].apply(trial_plotter,
                                      args=[StaticTorsoSegment.TRACKING_MARKERS,
                                            read_filling_directives(params.filling_directives), db.attrs['dt']])

    # create plots
    init_graphing()
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Outputting torso kinematics for subject %s', subject_name)
        subject_pdf_file = root_path / (subject_name + '.pdf')
        with PdfPages(subject_pdf_file) as subject_pdf:
            for plotter in subject_df['Plotter']:
                if plotter is not None:
                    log.info('Outputting torso kinematics for trial %s', plotter.trial_name)
                    figs = plotter.plot()
                    for fig in figs:
                        subject_pdf.savefig(fig)
                        fig.clf()
                        plt.close(fig)
