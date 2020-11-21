"""This module provides base classes and utilities for enabling easy access to the Vicon and biplane fluoroscopy
filesystem-based database."""

from pathlib import Path
import pandas as pd
import numpy as np
from lazy import lazy
import quaternion
from typing import Sequence, Union, Callable, Tuple
from ..kinematics.cs import ht_r

ACTIVITY_TYPES = ['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'IRaB', 'IRaM', 'Static', 'WCA', 'WSA', 'WFE']
"""Short code that appears in the trial name for the activities that subjects performed."""

MARKERS = ['STRN', 'C7', 'T5', 'T10', 'LSHO', 'LCLAV', 'CLAV', 'RCLAV', 'RSH0', 'RACRM', 'RANGL', 'RSPIN', 'RUPAA',
           'RUPAB', 'RUPAC', 'RUPAD', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RHNDA', 'RHNDB', 'RHNDC', 'RHNDD']
"""Skin markers affixed to subjects, named after the anatomical landmark they represent."""


class ViconEndpts:
    """Indices (endpoints) that indicate which of the frames in a Vicon trial correspond to the endpoints of the
    reciprocal biplane fluoroscopy trial.

    Provides lazy (and cached) access to the endpoints. Designed as a mix-in class.

    Attributes
    ----------
    endpts_file: Path
        Path to the CSV file containing the endpoint frame indices.
    """

    def __init__(self, endpts_file: Union[Path, Callable], **kwargs):
        if callable(endpts_file):
            self.endpts_file = endpts_file()
        else:
            self.endpts_file = endpts_file
        assert (self.endpts_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def vicon_endpts(self) -> np.ndarray:
        """Indices (endpoints) that indicate which of the frames in a Vicon trial correspond to the endpoints of the
        reciprocal biplane fluoroscopy trial."""
        _endpts_df = pd.read_csv(self.endpts_file, header=0)
        vicon_endpts = np.squeeze(_endpts_df.to_numpy())
        # the exported values assume that the first vicon frame is 1 but Python uses 0 based indexing
        # the exported values are inclusive but most Python and numpy functions (arange) are exclusive of the stop
        # so that's why the stop value is not modified
        vicon_endpts[0] -= 1
        return vicon_endpts


class TrialDescription:
    """Description (name, activity, and number) of a trial in the Vicon and biplane fluoroscopy database.

    Designed as a mix-in class.

    Attributes
    ----------
    trial_name: str
        The unique trial identifier, e.g. N002A_CA_t01.
    subject_short
        The shortened subject identifer contained in the `trial_name`, e.g. N002A.
    activity
        The activity code contained in the `trial_name`, e.g. CA.
    trial_number
        The trial number contained in the `trial_name`, e.g. 1.
    """

    def __init__(self, trial_dir_path: Path, **kwargs):
        # now create trial identifiers
        self.trial_name = trial_dir_path.stem
        self.subject_short, self.activity, self.trial_number = TrialDescription.parse_trial_name(self.trial_name)
        super().__init__(**kwargs)

    @staticmethod
    def parse_trial_name(trial_name: str) -> Tuple[str, str, int]:
        trial_name_split = trial_name.split('_')
        subject = trial_name_split[0]
        activity = trial_name_split[1]
        trial_number = int(trial_name_split[2][1:])
        return subject, activity, trial_number


def trial_descriptor_df(subject_name: str, trials: Sequence[TrialDescription]) -> pd.DataFrame:
    """Return a Pandas dataframe that contains commonly used fields from the trials supplied."""
    return pd.DataFrame({'Subject_Name': pd.Series([subject_name] * len(trials), dtype=pd.StringDtype()),
                         'Trial_Name': pd.Series([trial.trial_name for trial in trials], dtype=pd.StringDtype()),
                         'Subject_Short': pd.Series([trial.subject_short for trial in trials], dtype=pd.StringDtype()),
                         'Activity': pd.Categorical([trial.activity for trial in trials], categories=ACTIVITY_TYPES),
                         'Trial_Number': pd.Series([trial.trial_number for trial in trials], dtype=np.int)})


class SubjectDescription:
    """Description (name) of a subject in the Vicon and biplane fluoroscopy database.

    Designed as a mix-in class.

    Attributes
    ----------
    subject_name: pathlib.Path
        The subject identifier.
    """

    def __init__(self, subject_dir_path: Path, **kwargs):
        # subject identifier
        self.subject_name = subject_dir_path.stem
        super().__init__(**kwargs)


class ViconCSTransform:
    """Vicon to biplane fluoroscopy homogeneous coordinate system (CS) transformation for a subject in the Vicon and
    biplane fluoroscopy database.

    All trials for a subject utilize the same CS transformation. Designed as a mix-in class.

    Attributes
    ----------
    f_t_v_file: pathlib.Path
        Path to the file containing the Vicon to biplane fluoroscopy coordinate system transformation data.
    """

    def __init__(self, f_t_v_file: Union[Callable, Path], **kwargs):
        if callable(f_t_v_file):
            self.f_t_v_file = f_t_v_file()
        else:
            self.f_t_v_file = f_t_v_file
        assert(self.f_t_v_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def f_t_v_data(self) -> pd.DataFrame:
        """Pandas dataframe of the CS transformation (expressed as a quaternion and translation) as read from the
        containing file."""
        return pd.read_csv(self.f_t_v_file, header=0)

    @lazy
    def f_t_v(self) -> np.ndarray:
        """Homogeneous CS transformation ((4, 4) numpy array) from the Vicon to the biplane fluoroscopy CS."""
        r = quaternion.as_rotation_matrix(quaternion.from_float_array(self.f_t_v_data.iloc[0, :4].to_numpy()))
        return ht_r(r, self.f_t_v_data.iloc[0, 4:].to_numpy())
