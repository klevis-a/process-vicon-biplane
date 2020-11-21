"""Graph scapula/torso kinematics for every trial in the biplane/Vicon filesystem-based database.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing exported biplane fluoroscopy data
output_dir: Path to where PDF file summaries should be created
"""


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from biplane_kine.kinematics.euler_angles import st_isb
    from biplane_kine.smoothing.mean_smoother import smooth_trajectory
    from biplane_kine.kinematics.cs import change_cs
    from biplane_kine.kinematics.trajectory import PoseTrajectory
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import BiplaneViconSubject
    from biplane_kine.graphing.kine_plotters import RawSmoothSegmentPlotter
    from biplane_kine.misc.json_utils import Params
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from ..general.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(mod_arg_parser('Batch plot scapula/humerus kinematics', __package__, __file__))
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)

    # select trial and extract needed data
    trial = db.loc[params.trial_name, 'Trial']
    scap_biplane_data = trial.scapula_biplane_data
    hum_biplane_data = trial.humerus_biplane_data

    # create raw trajectories
    # scapula
    scap_quat = scap_biplane_data.iloc[:, 3:].to_numpy()
    scap_pos = scap_biplane_data.iloc[:, :3].to_numpy()
    scap_raw = PoseTrajectory.from_quat(scap_pos, scap_quat, db.attrs['dt'])
    # humerus
    hum_quat = hum_biplane_data.iloc[:, 3:].to_numpy()
    hum_pos = hum_biplane_data.iloc[:, :3].to_numpy()
    hum_raw = PoseTrajectory.from_quat(hum_pos, hum_quat, db.attrs['dt'])
    assert(np.array_equal(scap_biplane_data.index.to_numpy(), hum_biplane_data.index.to_numpy()))
    frame_nums = scap_biplane_data.index.to_numpy()
    # torso
    torso_vicon = trial.torso_vicon_v3d[trial.vicon_endpts[0]:trial.vicon_endpts[1]][frame_nums-1]
    torso = PoseTrajectory.from_ht(change_cs(trial.subject.f_t_v, torso_vicon), db.attrs['dt'])

    # smooth
    scap_smooth = smooth_trajectory(scap_raw, params.num_frames_avg)
    hum_smooth = smooth_trajectory(hum_raw, params.num_frames_avg)

    # express in the first torso frame (I don't want to express in torso because that will add noise)
    scap_raw_torso = scap_raw.in_frame(torso.ht[0])
    hum_raw_torso = hum_raw.in_frame(torso.ht[0])
    scap_smooth_torso = scap_smooth.in_frame(torso.ht[0])
    hum_smooth_torso = hum_smooth.in_frame(torso.ht[0])

    # plot
    pos_legend = ['X', 'Y', 'Z']
    init_graphing()
    scap_plotter = \
        RawSmoothSegmentPlotter(trial.trial_name, 'Scapula', scap_raw_torso.pos,
                                np.rad2deg(scap_raw_torso.euler.st_isb), scap_smooth_torso.pos,
                                np.rad2deg(scap_smooth_torso.euler.st_isb), scap_smooth_torso.vel,
                                np.rad2deg(scap_smooth_torso.ang_vel), frame_nums, st_isb.legend_short, pos_legend)
    figs = scap_plotter.plot()
    with PdfPages(r'C:\Users\klevis\Desktop\test.pdf') as pdf:
        for idx in (0, 1, 5, 6, 7):
            fig = figs[idx]
            pdf.savefig(fig)
            fig.clf()
            plt.close(fig)
