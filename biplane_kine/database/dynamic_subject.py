from pathlib import Path
import pandas as pd
from ezc3d import c3d
from .c3d_helper import C3DHelper
from .dynamic_trial import DynamicTrial
from .db_common import SubjectDescriptor, ViconCSTransform, trial_descriptor_df


class DynamicSubject(SubjectDescriptor, ViconCSTransform):
    def __init__(self, subject_dir, **kwargs):
        self.subject_dir_path = subject_dir if isinstance(subject_dir, Path) else Path(subject_dir)
        self.static_dir = self.subject_dir_path / 'Static'
        super().__init__(subject_dir_path=self.subject_dir_path, f_t_v_file=self.static_dir / 'F_T_V.csv', **kwargs)

        # file paths
        self.humerus_stl_file = self.static_dir / 'Humerus.stl'
        self.scapula_stl_file = self.static_dir / 'Scapula.stl'
        self.humerus_landmarks_file = self.static_dir / 'humerus_landmarks.csv'
        self.scapula_landmarks_file = self.static_dir / 'scapula_landmarks.csv'
        self.static_c3d_file = self.static_dir / 'vicon_static_trial.c3d'
        self.static_csv_file = self.static_dir / 'vicon_static_trial.csv'

        # make sure the files are actually there
        assert (self.humerus_stl_file.is_file())
        assert (self.scapula_stl_file.is_file())
        assert (self.humerus_landmarks_file.is_file())
        assert (self.scapula_landmarks_file.is_file())
        assert (self.static_c3d_file.is_file())
        assert (self.static_csv_file.is_file())

        # create variables that are empty so initialization is lazy
        self._static_c3d_helper = None
        self._static_vicon_data = None
        self._df = None

        # dynamic trials
        self.trials = [DynamicTrial(trial_dir) for trial_dir in self.subject_dir_path.iterdir() if
                       (trial_dir.is_dir() and trial_dir.name != 'Static')]

        # used for dataframe
        self._df = None

    @property
    def static_c3d_helper(self):
        if self._static_c3d_helper is None:
            self._static_c3d_helper = C3DHelper(c3d(str(self.static_c3d_file)))
        return self._static_c3d_helper

    @property
    def static_vicon_data(self):
        if self._static_vicon_data is None:
            self._static_vicon_data = pd.read_csv(self.static_csv_file, header=[0, 1])
        return self._static_vicon_data

    @property
    def subject_df(self):
        if self._df is None:
            self._df = trial_descriptor_df(self.subject_name, self.trials)
            self._df['Trial'] = pd.Series(self.trials, dtype=object)
        return self._df
