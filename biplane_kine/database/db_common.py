import pandas as pd
import numpy as np
from lazy import lazy
from scipy.spatial.transform import Rotation
from ..kinematics.cs import ht_r

ACTIVITY_TYPES = ['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'IRaB', 'IRaM', 'Static', 'WCA', 'WSA', 'WFE']
MARKERS = ['STRN', 'C7', 'T5', 'T10', 'LSHO', 'LCLAV', 'CLAV', 'RCLAV', 'RSH0', 'RACRM', 'RANGL', 'RSPIN', 'RUPAA',
           'RUPAB', 'RUPAC', 'RUPAD', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RHNDA', 'RHNDB', 'RHNDC', 'RHNDD']


def trial_descriptor_df(subject_name, trials):
    return pd.DataFrame({'Subject_Name': pd.Series([subject_name] * len(trials), dtype=pd.StringDtype()),
                         'Trial_Name': pd.Series([trial.trial_name for trial in trials], dtype=pd.StringDtype()),
                         'Subject_Short': pd.Series([trial.subject_short for trial in trials], dtype=pd.StringDtype()),
                         'Activity': pd.Categorical([trial.activity for trial in trials], categories=ACTIVITY_TYPES),
                         'Trial_Number': pd.Series([trial.trial_number for trial in trials], dtype=np.int)})


class ViconEndpts:
    def __init__(self, endpts_file, **kwargs):
        if callable(endpts_file):
            self.endpts_file = endpts_file()
        else:
            self.endpts_file = endpts_file
        assert (self.endpts_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def vicon_endpts(self):
        endpts_df = pd.read_csv(self.endpts_file, header=0)
        vicon_endpts = np.squeeze(endpts_df.to_numpy())
        # the exported values assume that the first vicon frame is 1 but Python uses 0 based indexing
        # the exported values are inclusive but most Python and numpy functions (arange) are exclusive of the stop
        # so that's why the stop value is not modified
        vicon_endpts[0] -= 1
        return vicon_endpts


class TrialDescriptor:
    def __init__(self, trial_dir_path, **kwargs):
        # now create trial identifiers
        self.trial_name = trial_dir_path.stem
        self.subject_short, self.activity, self.trial_number = TrialDescriptor.parse_trial_name(self.trial_name)
        super().__init__(**kwargs)

    @staticmethod
    def parse_trial_name(trial_name):
        trial_name_split = trial_name.split('_')
        subject = trial_name_split[0]
        activity = trial_name_split[1]
        trial_number = int(trial_name_split[2][1:])
        return subject, activity, trial_number


class SubjectDescriptor:
    def __init__(self, subject_dir_path, **kwargs):
        # subject identifier
        self.subject_name = subject_dir_path.stem
        super().__init__(**kwargs)


class ViconCSTransform:
    def __init__(self, f_t_v_file, **kwargs):
        if callable(f_t_v_file):
            self.f_t_v_file = f_t_v_file()
        else:
            self.f_t_v_file = f_t_v_file
        assert(self.f_t_v_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def f_t_v_data(self):
        return pd.read_csv(self.f_t_v_file, header=0)

    @lazy
    def f_t_v(self):
        q_imp = self.f_t_v_data.iloc[0, :4].to_numpy()
        # convert to scalar last format
        q = np.concatenate((q_imp[1:], [q_imp[0]]))
        r = Rotation.from_quat(q)
        return ht_r(r.as_matrix(), self.f_t_v_data.iloc[0, 4:].to_numpy())
