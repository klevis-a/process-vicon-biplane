import pandas as pd
import numpy as np

ACTIVITY_TYPES = ['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'IRaB', 'IRaM', 'Static', 'WCA', 'WSA', 'WFE']
MARKERS = ['T10', 'T5', 'C7', 'STRN', 'CLAV', 'LSHO', 'LCLAV', 'RCLAV', 'RSH0', 'RACRM', 'RSPIN', 'RANGL', 'RUPAA',
           'RUPAB', 'RUPAC', 'RUPAD', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RHNDA', 'RHNDB', 'RHNDC', 'RHNDD']


class ViconEndpts:
    def __init__(self, endpts_file, **kwargs):
        super().__init__(**kwargs)
        self.endpts_file = endpts_file
        assert (self.endpts_file.is_file())
        self._vicon_endpts = None

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


class TrialDescriptor:
    def __init__(self, trial_dir_path, **kwargs):
        super().__init__(**kwargs)
        # now create trial identifiers
        self.trial_name = trial_dir_path.stem
        self.subject_short, self.activity, self.trial_number = TrialDescriptor.parse_trial_name(self.trial_name)

    @staticmethod
    def parse_trial_name(trial_name):
        trial_name_split = trial_name.split('_')
        subject = trial_name_split[0]
        activity = trial_name_split[1]
        trial_number = int(trial_name_split[2][1:])
        return subject, activity, trial_number


class SubjectDescriptor:
    def __init__(self, subject_dir_path, **kwargs):
        super().__init__(**kwargs)
        # subject identifier
        self.subject = subject_dir_path.stem


class SubjectDf:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._df = None

    @property
    def subject_df(self):
        if self._df is None:
            self._df = pd.DataFrame({'Subject': self.subject,
                                     'Trial_Name': [trial.trial_name for trial in self.trials],
                                     'Subject_Short': [trial.subject_short for trial in self.trials],
                                     'Activity': pd.Categorical([trial.activity for trial in self.trials],
                                                                categories=ACTIVITY_TYPES),
                                     'Trial_number': [trial.trial_number for trial in self.trials],
                                     'Trial': self.trials})
        return self._df


class ViconCSTransform:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._F_T_V_data = None

    @property
    def f_t_v_data(self):
        if self._F_T_V_data is None:
            self._F_T_V_data = pd.read_csv(self.F_T_V_file, header=0)
        return self._F_T_V_data
