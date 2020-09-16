import sys
import matplotlib.pyplot as plt
from database import create_db
from parameters import Params
import graphing.graph_utils as graph
from graphing.plotters import SmoothingDebugPlotter
from smoothing.kf_filtering_helpers import kf_filter_marker_all, post_process_raw

# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)

# filter
trial = db.loc[params.trial_name].Trial
raw, filled = post_process_raw(trial, params.marker_name, dt=db.attrs['dt'])
filtered, smoothed = kf_filter_marker_all(trial, params.marker_name, dt=db.attrs['dt'])

# graphing
graph.init_graphing()
marker_plotter = SmoothingDebugPlotter(trial.trial_name, params.marker_name, raw, filtered, smoothed,
                                       trial.vicon_endpts)
marker_plotter.plot()
plt.show()
