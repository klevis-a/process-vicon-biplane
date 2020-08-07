from pathlib import Path
import numpy as np
import pandas as pd
from ezc3d import c3d
from .c3d_helper import C3DHelper


class DynamicTrial:
    BIPLANE_FILE_HEADERS = {'frame': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                            'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}

    ACTIVITY_TYPES = ['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'IRaB', 'IRaM', 'Static', 'WCA', 'WSA', 'WFE']

    def __init__(self, trial_dir):

        # file paths
        if isinstance(trial_dir, Path):
            self.trial_dir_path = trial_dir
        else:
            self.trial_dir_path = Path(trial_dir)

        self.humerus_biplane_file = self.trial_dir_path / 'humerus_biplane.csv'
        self.scapula_biplane_file = self.trial_dir_path / 'scapula_biplane.csv'
        self.endpts_file = self.trial_dir_path / 'vicon_endpts.csv'
        self.c3d_file = self.trial_dir_path / (self.trial_dir_path.name + '.c3d')
        self.vicon_csv_file = self.trial_dir_path / (self.trial_dir_path.name + '.csv')

        # make sure the files are actually there
        assert (self.humerus_biplane_file.is_file())
        assert (self.scapula_biplane_file.is_file())
        assert (self.endpts_file.is_file())
        assert (self.c3d_file.is_file())

        # create variables that are empty so initialization is lazy
        self._humerus_biplane_data = None
        self._scapula_biplane_data = None
        self._c3d_helper = None
        self._vicon_endpts = None
        self._vicon_data = None

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
    def c3d_helper(self):
        if self._c3d_helper is None:
            self._c3d_helper = C3DHelper(c3d(str(self.c3d_file)))
        return self._c3d_helper

    @property
    def vicon_data(self):
        if self._vicon_data is None:
            self._vicon_data = pd.read_csv(self.vicon_csv_file, header=[0, 1])
        return self._vicon_data

    def marker_data_df(self, marker_name):
        return self.vicon_data[marker_name]

    def marker_data(self, marker_name, replace_nan=False):
        if replace_nan:
            return self.marker_data_df(marker_name).replace({np.nan: None}).values
        else:
            return self.marker_data_df(marker_name).values
