from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
from .db_common import ViconEndpts, MARKERS, TrialDescriptor, SubjectDescriptor, ViconCSTransform, trial_descriptor_df

ViconAccuracyMarkerData = namedtuple('ViconAccuracyMarkerData', ['indices', 'frames', 'data'])
VICON_ACCURACY_FILE_HEADERS = {'frame': np.int32, 'x': np.float64, 'y': np.float64, 'z': np.float64}


class ViconAccuracyTrial(TrialDescriptor):
    def __init__(self, trial_dir, subject, **kwargs):
        self.trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        super().__init__(trial_dir_path=self.trial_dir_path, **kwargs)

        self.subject = subject
        self._marker_files = {file.stem: file for file in self.trial_dir_path.iterdir() if
                              (file.stem in MARKERS)}
        self._marker_data = {k: ViconAccuracyTrial.process_marker_file(v) for k, v in self._marker_files.items()}
        self.markers = self._marker_files.keys()

    @staticmethod
    def read_marker_file(file_path):
        return pd.read_csv(file_path, header=0, dtype=VICON_ACCURACY_FILE_HEADERS, index_col='frame')

    @staticmethod
    def process_marker_file(file_path):
        marker_data = ViconAccuracyTrial.read_marker_file(file_path)
        frames = marker_data.index.to_numpy()
        return ViconAccuracyMarkerData(frames - 1, frames, marker_data.to_numpy())

    def __getitem__(self, marker_name):
        return self._marker_data[marker_name]

    def __iter__(self):
        return (i for i in self._marker_data.items())


class ViconAccuracyTrialEndpts(ViconAccuracyTrial, ViconEndpts):
    def __init__(self, trial_dir, subject):
        trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        super().__init__(trial_dir=trial_dir_path, subject=subject, endpts_file=trial_dir_path / 'vicon_endpts.csv')


class ViconAccuracySubjectEndpts(SubjectDescriptor, ViconCSTransform):
    def __init__(self, subj_dir, **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        super().__init__(subject_dir_path=self.subject_dir_path, f_t_v_file=self.subject_dir_path / 'F_T_V.csv',
                         **kwargs)
        self.trials = [ViconAccuracyTrialEndpts(folder, self) for folder in self.subject_dir_path.iterdir()
                       if folder.is_dir()]
        # used for dataframe
        self._df = None

    @property
    def subject_df(self):
        if self._df is None:
            self._df = trial_descriptor_df(self.subject_name, self.trials)
            self._df['Accuracy_Trial'] = pd.Series(self.trials, dtype=object)
        return self._df
