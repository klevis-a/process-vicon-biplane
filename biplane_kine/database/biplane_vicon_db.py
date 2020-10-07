from pathlib import Path
import numpy as np
import pandas as pd
from lazy import lazy
from .db_common import TrialDescriptor, ViconEndpts, SubjectDescriptor, trial_descriptor_df
from ..misc.python_utils import NestedContainer


def csv_get_item_method(csv_data, marker_name):
    return csv_data.loc[:, marker_name:(marker_name + '.2')].to_numpy()


class ViconCsvTrial(TrialDescriptor, ViconEndpts):
    BIPLANE_FILE_HEADERS = {'frame': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                            'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}

    def __init__(self, trial_dir, **kwargs):
        self.trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        super().__init__(trial_dir_path=self.trial_dir_path,
                         endpts_file=lambda: self.trial_dir_path / (self.trial_name + '_vicon_endpts.csv'), **kwargs)

        # file paths
        self.vicon_csv_file_labeled = self.trial_dir_path / (self.trial_name + '_vicon_labeled.csv')
        self.vicon_csv_file_filled = self.trial_dir_path / (self.trial_name + '_vicon_filled.csv')

        # make sure the files are actually there
        assert (self.vicon_csv_file_labeled.is_file())
        assert (self.vicon_csv_file_filled.is_file())

    @lazy
    def vicon_csv_data_labeled(self):
        # TODO: this works fine for now and by using the accessor method below we get a view (rather than a copy) of the
        #  data, however it probably makes sense to using something like structured arrays or xarray. Note that
        #  multi-level column labels should not be used (i.e. header=[0, 1) because a copy of the data, not a view is
        #  returned
        return pd.read_csv(self.vicon_csv_file_labeled, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def vicon_csv_data_filled(self):
        return pd.read_csv(self.vicon_csv_file_filled, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def labeled(self):
        return NestedContainer(self.vicon_csv_data_labeled, csv_get_item_method)

    @lazy
    def filled(self):
        return NestedContainer(self.vicon_csv_data_filled, csv_get_item_method)


class ViconCsvSubject(SubjectDescriptor):
    def __init__(self, subj_dir, **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        super().__init__(subject_dir_path=self.subject_dir_path, **kwargs)
        self.trials = [ViconCsvTrial(folder) for folder in self.subject_dir_path.iterdir() if (folder.is_dir() and
                       folder.stem != 'Static')]

    @lazy
    def subject_df(self):
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Trial'] = pd.Series(self.trials, dtype=object)
        return df
