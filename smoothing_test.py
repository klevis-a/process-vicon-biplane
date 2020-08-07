import sys
import numpy as np
import graphing.graph_utils as graph
from database import create_db
from parameters import Params
from smoothing.kalman_filtering import LinearKalmanFilter1D

# ready db
params = Params.get_params(sys.argv[1])
db_dir = params.db_dir
db, anthro = create_db(db_dir)

# kalman filter
trial_name = 'N002A_CA_t01'
marker_name = 'RSH0'
trial = db.loc[trial_name].Trial
dt = 1 / 100
kf = LinearKalmanFilter1D(dt=dt, discrete_white_noise_var=10000, r=1, p=np.diag([0.5, 0.5, 0.5]), db=db)
marker_data, marker_data_filtered, marker_data_smoothed = kf.filter_trial_marker(trial.marker_data(marker_name,
                                                                                                   replace_nan=True))

# graphing
graph.init_graphing()
_, ax, x_data = graph.marker_graph_init(marker_data, 'Marker data for ' + marker_name, fig_num=0,
                                        x_data=np.arange(30, 330))
graph.marker_graph_add(ax, marker_data_filtered, x_data, 'r-')
graph.marker_graph_add(ax, marker_data_smoothed, x_data, 'g-')
graph.make_interactive()
