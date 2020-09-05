import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from database import create_db
from parameters import Params
from smoothing.misc_smoothing import x0_guess
from smoothing.kalman_filtering import LinearKF1DSimdKalman
import graphing.graph_utils as graph
from graphing.plotters import SmoothingOutputPlotter

# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)

# filter
trial = db.loc[params.trial_name].Trial
marker_pos_labeled = trial.marker_data_labeled(params.marker_name)
marker_pos_filled = trial.marker_data_filled(params.marker_name)
x0, start_idx, stop_idx = x0_guess(marker_pos_labeled, marker_pos_filled, db.attrs['dt'], 50, 10)
p = np.tile(np.diag([1, 100, 1000])[:, :, np.newaxis], 3)
kf = LinearKF1DSimdKalman(dt=db.attrs['dt'], discrete_white_noise_var=10000, r=1)
filtered_means, smoothed_means, filtered_covs, smoothed_covs = \
    kf.filter_trial_marker(marker_pos_labeled[start_idx:stop_idx, :], x0, p)

# post process
FilterStep = namedtuple('FilterStep', ['endpts', 'indices', 'means', 'covars', 'corrs'])
marker_vel_labeled = np.gradient(marker_pos_labeled, db.attrs['dt'], axis=0)
marker_acc_labeled = np.gradient(marker_vel_labeled, db.attrs['dt'], axis=0)
raw_means = LinearKF1DSimdKalman.StateMeans(marker_pos_labeled, marker_vel_labeled, marker_acc_labeled)
marker_vel_filled = np.gradient(marker_pos_filled, db.attrs['dt'], axis=0)
marker_acc_filled = np.gradient(marker_vel_filled, db.attrs['dt'], axis=0)
filled_means = LinearKF1DSimdKalman.StateMeans(marker_pos_filled, marker_vel_filled, marker_acc_filled)
filtered_corrs = LinearKF1DSimdKalman.CorrVec(*LinearKF1DSimdKalman.extract_corrs(filtered_covs))
smoothed_corrs = LinearKF1DSimdKalman.CorrVec(*LinearKF1DSimdKalman.extract_corrs(smoothed_covs))

raw_endpts = (0, marker_pos_labeled.shape[0])
raw_indices = np.arange(raw_endpts[1])
filtered_endpts = (start_idx, stop_idx)
filtered_indices = np.arange(filtered_endpts[0], filtered_endpts[1])

raw = FilterStep(raw_endpts, raw_indices, raw_means, None, None)
filled = FilterStep(raw_endpts, raw_indices, filled_means, None, None)
filtered = FilterStep(filtered_endpts, filtered_indices, filtered_means, filtered_covs, filtered_corrs)
smoothed = FilterStep(filtered_endpts, filtered_indices, smoothed_means, smoothed_covs, smoothed_corrs)

# graphing
graph.init_graphing()
marker_plotter = SmoothingOutputPlotter(trial.trial_name, params.marker_name, raw, filled, filtered, smoothed,
                                        trial.vicon_endpts)
marker_plotter.plot()
plt.show()
