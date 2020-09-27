from pathlib import Path
import numpy as np
import pandas as pd
from .c3d_helper import C3DHelper
from .db_common import ViconEndpts, TrialDescriptor


class DynamicTrial(ViconEndpts, TrialDescriptor):
    BIPLANE_FILE_HEADERS = {'frame': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                            'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}

    def __init__(self, trial_dir, **kwargs):
        self.trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        super().__init__(endpts_file=self.trial_dir_path / 'vicon_endpts.csv', trial_dir_path=self.trial_dir_path,
                         **kwargs)

        # file paths
        self.humerus_biplane_file = self.trial_dir_path / 'humerus_biplane.csv'
        self.scapula_biplane_file = self.trial_dir_path / 'scapula_biplane.csv'
        self.c3d_file_labeled = self.trial_dir_path / (self.trial_dir_path.name + '.c3d')
        self.vicon_csv_file_labeled = self.trial_dir_path / (self.trial_dir_path.name + '.csv')
        self.c3d_file_filled = self.trial_dir_path / (self.trial_dir_path.name + '_filled.c3d')
        self.vicon_csv_file_filled = self.trial_dir_path / (self.trial_dir_path.name + '_filled.csv')

        # make sure the files are actually there
        assert (self.humerus_biplane_file.is_file())
        assert (self.scapula_biplane_file.is_file())
        assert (self.c3d_file_labeled.is_file())
        assert (self.vicon_csv_file_labeled.is_file())
        assert (self.c3d_file_filled.is_file())
        assert (self.vicon_csv_file_filled.is_file())

        # create variables that are empty so initialization is lazy
        self._humerus_biplane_data = None
        self._scapula_biplane_data = None
        self._c3d_helper_labeled = None
        self._vicon_data_labeled = None
        self._c3d_helper_filled = None
        self._vicon_data_filled = None

    @staticmethod
    def read_biplane_file(file_path):
        return pd.read_csv(file_path, header=0, dtype=DynamicTrial.BIPLANE_FILE_HEADERS, index_col='frame')

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
            self._c3d_helper_labeled = C3DHelper(str(self.c3d_file_labeled))
        return self._c3d_helper_labeled

    @property
    def c3d_helper_filled(self):
        if self._c3d_helper_filled is None:
            self._c3d_helper_filled = C3DHelper(str(self.c3d_file_filled))
        return self._c3d_helper_filled

    @property
    def vicon_data_labeled(self):
        # TODO: this works fine for now and by using the accessor method below we get a view (rather than a copy) of the
        #  data, however it probably makes sense to using something like structured arrays or xarray. Note that
        #  multi-level column labels should not be used (i.e. header=[0, 1) because a copy of the data, not a view is
        #  returned
        if self._vicon_data_labeled is None:
            self._vicon_data_labeled = pd.read_csv(self.vicon_csv_file_labeled, header=[0], skiprows=[1],
                                                   dtype=np.float64)
        return self._vicon_data_labeled

    def marker_data_labeled_df(self, marker_name):
        return self.vicon_data_labeled.loc[:, marker_name:(marker_name+'.2')]

    def marker_data_labeled(self, marker_name):
        return self.marker_data_labeled_df(marker_name).to_numpy()

    @property
    def vicon_data_filled(self):
        if self._vicon_data_filled is None:
            self._vicon_data_filled = pd.read_csv(self.vicon_csv_file_filled, header=[0], skiprows=[1],
                                                  dtype=np.float64)
        return self._vicon_data_filled

    def marker_data_filled_df(self, marker_name):
        return self.vicon_data_filled.loc[:, marker_name:(marker_name+'.2')]

    def marker_data_filled(self, marker_name):
        return self.marker_data_filled_df(marker_name).to_numpy()
