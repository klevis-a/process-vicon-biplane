import sys
import matplotlib.pyplot as plt
import graphing.graph_utils as graph
from database import create_db
from parameters import Params

# ready db
params = Params.get_params(sys.argv[1])
db, anthro = create_db(params.db_dir)
graph.init_graphing()
marker_plotter = graph.MarkerPlotter(db, params.trial_name, params.marker_name)
plt.show()
