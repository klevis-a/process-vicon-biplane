from pathlib import Path
import pandas as pd
from ezc3d import c3d
from .c3d_helper import C3DHelper
from .dynamic_trial import DynamicTrial


class Subject:
    def __init__(self, subject_dir):

        # file paths
        if isinstance(subject_dir, Path):
            self.subject_dir_path = subject_dir
        else:
            self.subject_dir_path = Path(subject_dir)

        # file paths
        self.static_dir = self.subject_dir_path / 'Static'
        self.humerus_stl_file = self.static_dir / 'Humerus.stl'
        self.scapula_stl_file = self.static_dir / 'Scapula.stl'
        self.humerus_landmarks_file = self.static_dir / 'humerus_landmarks.csv'
        self.scapula_landmarks_file = self.static_dir / 'scapula_landmarks.csv'
        self.static_c3d_file = self.static_dir / 'vicon_static_trial.c3d'
        self.static_csv_file = self.static_dir / 'vicon_static_trial.csv'
        self.F_T_V_file = self.static_dir / 'F_T_V.csv'

        # make sure the files are actually there
        assert (self.humerus_stl_file.is_file())
        assert (self.scapula_stl_file.is_file())
        assert (self.humerus_landmarks_file.is_file())
        assert (self.scapula_landmarks_file.is_file())
        assert (self.static_c3d_file.is_file())
        assert (self.static_csv_file.is_file())
        assert (self.F_T_V_file.is_file())

        # create variables that are empty so initialization is lazy
        self._static_c3d_helper = None
        self._static_vicon_data = None
        self._F_T_V_data = None

        # dynamic trials
        self.dynamic_trials = [DynamicTrial(trial_dir) for trial_dir in self.subject_dir_path.iterdir() if
                               (trial_dir.is_dir() and trial_dir.name != 'Static')]

        # subject identifier
        self.subject = self.subject_dir_path.stem

    @classmethod
    def create_subject_df(cls, subject):
        return pd.DataFrame({'Subject': subject.subject,
                             'Trial_Name': [trial.trial_name for trial in subject.dynamic_trials],
                             'Subject_Short': [trial.subject_short for trial in subject.dynamic_trials],
                             'Activity': pd.Categorical([trial.activity for trial in subject.dynamic_trials],
                                                        categories=DynamicTrial.ACTIVITY_TYPES),
                             'Trial_number': [trial.trial_number for trial in subject.dynamic_trials],
                             'Trial': subject.dynamic_trials})

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
    def f_t_v_data(self):
        if self._F_T_V_data is None:
            self._F_T_V_data = pd.read_csv(self.F_T_V_file, header=0)
        return self._F_T_V_data
