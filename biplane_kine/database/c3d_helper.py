import numpy as np
from ezc3d import c3d


class C3DHelper:
    def __init__(self, c3d_path):

        # Note that we extract the relevant parameters from c3d_obj and then it should be garbage collected -
        # this results in a significant savings in memory.
        c3d_obj = c3d(c3d_path)
        self.marker_names = c3d_obj['parameters']['POINT']['LABELS']['value']
        self.marker_map = dict(zip(self.marker_names, range(len(self.marker_names))))
        self.analog_frame_rate = c3d_obj['header']['analogs']['frame_rate']
        self.marker_frame_rate = c3d_obj['header']['points']['frame_rate']
        self.marker_data = c3d_obj['data']['points']
        self.frames = np.arange(self.marker_data.shape[2])

    def data_for_marker(self, marker_name):
        return self.marker_data[0:3, self.marker_map[marker_name], :].T
