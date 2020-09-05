import sys
import matplotlib.pyplot as plt
from database import create_db
from parameters import Params
import graphing.graph_utils as graph
from graphing.plotters import SmoothingOutputPlotter
from smoothing.kf_filtering_helpers import kf_filter_marker

# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)

# filter
trial = db.loc[params.trial_name].Trial
raw, filled, filtered, smoothed = kf_filter_marker(trial, params.marker_name, dt=db.attrs['dt'])

# graphing
graph.init_graphing()
marker_plotter = SmoothingOutputPlotter(trial.trial_name, params.marker_name, raw, filled, filtered, smoothed,
                                        trial.vicon_endpts)
marker_plotter.plot()
plt.show()
