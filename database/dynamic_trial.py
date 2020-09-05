from pathlib import Path
import numpy as np
import pandas as pd
from ezc3d import c3d
from .c3d_helper import C3DHelper


class DynamicTrial:
    BIPLANE_FILE_HEADERS = {'frame': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                            'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}

    ACTIVITY_TYPES = ['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'IRaB', 'IRaM', 'Static', 'WCA', 'WSA', 'WFE']

    MARKERS = ['T10', 'T5', 'C7', 'STRN', 'CLAV', 'LSHO', 'LCLAV', 'RCLAV', 'RSH0', 'RACRM', 'RSPIN', 'RANGL', 'RUPAA',
               'RUPAB', 'RUPAC', 'RUPAD', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RHNDA', 'RHNDB', 'RHNDC', 'RHNDD']

    def __init__(self, trial_dir):

        # file paths
        if isinstance(trial_dir, Path):
            self.trial_dir_path = trial_dir
        else:
            self.trial_dir_path = Path(trial_dir)

        self.humerus_biplane_file = self.trial_dir_path / 'humerus_biplane.csv'
        self.scapula_biplane_file = self.trial_dir_path / 'scapula_biplane.csv'
        self.endpts_file = self.trial_dir_path / 'vicon_endpts.csv'
        self.c3d_file_labeled = self.trial_dir_path / (self.trial_dir_path.name + '.c3d')
        self.vicon_csv_file_labeled = self.trial_dir_path / (self.trial_dir_path.name + '.csv')
        self.c3d_file_filled = self.trial_dir_path / (self.trial_dir_path.name + '_filled.c3d')
        self.vicon_csv_file_filled = self.trial_dir_path / (self.trial_dir_path.name + '_filled.csv')

        # make sure the files are actually there
        assert (self.humerus_biplane_file.is_file())
        assert (self.scapula_biplane_file.is_file())
        assert (self.endpts_file.is_file())
        assert (self.c3d_file_labeled.is_file())
        assert (self.vicon_csv_file_labeled.is_file())
        assert (self.c3d_file_filled.is_file())
        assert (self.vicon_csv_file_filled.is_file())

        # create variables that are empty so initialization is lazy
        self._humerus_biplane_data = None
        self._scapula_biplane_data = None
        self._vicon_endpts = None
        self._c3d_helper_labeled = None
        self._vicon_data_labeled = None
        self._c3d_helper_filled = None
        self._vicon_data_filled = None

        # now create trial identifiers
        self.trial_name = self.trial_dir_path.stem
        self.subject_short, self.activity, self.trial_number = DynamicTrial.parse_trial_name(self.trial_name)

    @classmethod
    def read_biplane_file(cls, file_path):
        return pd.read_csv(file_path, header=0, dtype=DynamicTrial.BIPLANE_FILE_HEADERS, index_col='frame')

    @classmethod
    def parse_trial_name(cls, trial_name):
        trial_name_split = trial_name.split('_')
        subject = trial_name_split[0]
        activity = trial_name_split[1]
        trial_number = int(trial_name_split[2][1:])
        return subject, activity, trial_number

    @property
    def humerus_biplane_data(self):
        if self._humerus_biplane_data is None:
            self._humerus_biplane_data = DynamicTrial.read_biplane_file(self.humerus_biplane_file)
        return self._humerus_biplane_data

    @property
    def scapula_biplane_data(self):
        if self._scapula_biplane_data is None:
            self._scapula_biplane_data = DynamicTrial.read_biplane_file(self.scapula_biplane_file)
        return self._scapula_biplane_data

    @property
    def c3d_helper_labeled(self):
        if self._c3d_helper_labeled is None:
            self._c3d_helper_labeled = C3DHelper(c3d(str(self.c3d_file_labeled)))
        return self._c3d_helper_labeled

    @property
    def c3d_helper_filled(self):
        if self._c3d_helper_filled is None:
            self._c3d_helper_filled = C3DHelper(c3d(str(self.c3d_file_filled)))
        return self._c3d_helper_filled

    @property
    def vicon_data_labeled(self):
        if self._vicon_data_labeled is None:
            self._vicon_data_labeled = pd.read_csv(self.vicon_csv_file_labeled, header=[0, 1])
        return self._vicon_data_labeled

    def marker_data_labeled_df(self, marker_name):
        return self.vicon_data_labeled[marker_name]

    def marker_data_labeled(self, marker_name):
        return self.marker_data_labeled_df(marker_name).to_numpy()

    @property
    def vicon_data_filled(self):
        if self._vicon_data_filled is None:
            self._vicon_data_filled = pd.read_csv(self.vicon_csv_file_filled, header=[0, 1])
        return self._vicon_data_filled

    def marker_data_filled_df(self, marker_name):
        return self.vicon_data_filled[marker_name]

    def marker_data_filled(self, marker_name):
        return self.marker_data_filled_df(marker_name).to_numpy()

    @property
    def vicon_endpts(self):
        if self._vicon_endpts is None:
            endpts_df = pd.read_csv(self.endpts_file, header=0)
            self._vicon_endpts = np.squeeze(endpts_df.to_numpy())
            # the exported values assume that the first vicon frame is 1 but Python uses 0 based indexing
            # the exported values are inclusive but most Python and numpy functions (arange) are exclusive of the stop
            # so that's why the stop value is not modified
            self._vicon_endpts[0] -= 1
        return self._vicon_endpts
