"""This module provides access to the Vicon and biplane fluoroscopy filesystem-based database."""

from pathlib import Path
import numpy as np
import pandas as pd
from lazy import lazy
from typing import Union, Callable
from .db_common import TrialDescription, ViconEndpts, SubjectDescription, ViconCSTransform, trial_descriptor_df
from ..misc.python_utils import NestedDescriptor


def csv_get_item_method(csv_data: pd.DataFrame, marker_name: str) -> np.ndarray:
    """Return the marker data, (n, 3) numpy array view, associated with marker_name."""
    return csv_data.loc[:, marker_name:(marker_name + '.2')].to_numpy()


def csv_get_item_method_squeeze(csv_data: pd.DataFrame, marker_name: str) -> np.ndarray:
    """Return the marker data, (n, 3) numpy array view, associated with marker_name."""
    return np.squeeze(csv_data.loc[:, marker_name:(marker_name + '.2')].to_numpy())


class ViconCsvTrial(TrialDescription, ViconEndpts):
    """A Vicon trial that has been exported to CSV format.

    Enables lazy (and cached) access to the labeled and filled Vicon Data.

    Attributes
    ----------
    trial_dir_path: pathlib.Path or str
        Path to the directory where the Vicon CSV trial data resides.
    vicon_csv_file_labeled: pathlib.Path
        Path to the labeled marker data for the Vicon CSV trial.
    vicon_csv_file_filled: pathlib.Path
        Path to the filled marker data for the Vicon CSV trial.
    """

    def __init__(self, trial_dir: Union[str, Path], **kwargs):
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
    def vicon_csv_data_labeled(self) -> pd.DataFrame:
        """Pandas dataframe with the labeled Vicon CSV data."""
        # TODO: this works fine for now and by using the accessor method below we get a view (rather than a copy) of the
        #  data, however it probably makes sense to using something like structured arrays or xarray. Note that
        #  multi-level column labels should not be used (i.e. header=[0, 1) because a copy of the data, not a view is
        #  returned
        return pd.read_csv(self.vicon_csv_file_labeled, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def vicon_csv_data_filled(self) -> pd.DataFrame:
        """Pandas dataframe with the filled Vicon CSV data."""
        return pd.read_csv(self.vicon_csv_file_filled, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def labeled(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to labeled CSV data. The indexed access returns
        a (n, 3) numpy array view."""
        return NestedDescriptor(self.vicon_csv_data_labeled, csv_get_item_method)

    @lazy
    def filled(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to filled CSV data. The indexed access return
        a (n, 3) numpy array view."""
        return NestedDescriptor(self.vicon_csv_data_filled, csv_get_item_method)


class BiplaneViconTrial(ViconCsvTrial):
    """A trial that contains both biplane and Vicon data.

    Attributes
    ----------
    vicon_csv_file_smoothed: pathlib.Path
        Path to the smoothed marker data for the Vicon CSV trial.
    subject: biplane_kine.database.vicon_accuracy.BiplaneViconSubject
        Pointer to the subject that contains this trial.
    """

    def __init__(self, trial_dir: Union[str, Path], subject: 'BiplaneViconSubject', **kwargs):
        super().__init__(trial_dir, **kwargs)
        self.subject = subject
        # file paths
        self.vicon_csv_file_smoothed = self.trial_dir_path / (self.trial_name + '_vicon_smoothed.csv')

        # make sure the files are actually there
        assert (self.vicon_csv_file_smoothed.is_file())

    @lazy
    def vicon_csv_data_smoothed(self) -> pd.DataFrame:
        """Pandas dataframe with the smoothed Vicon CSV data."""
        return pd.read_csv(self.vicon_csv_file_smoothed, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def smoothed(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to smoothed CSV data. The indexed access returns
        a (n, 3) numpy array view."""
        return NestedDescriptor(self.vicon_csv_data_smoothed, csv_get_item_method)


class ViconCsvSubject(SubjectDescription):
    """A subject that contains multiple Vicon CSV trials.

    Attributes
    ----------
    subject_dir_path: pathlib.Path
        Path to directory containing subject data.
    trials: list of biplane_kine.database.biplane_vicon_db.ViconCsvTrial
        List of trials for the subject.
    """

    def __init__(self, subj_dir: Union[str, Path], **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        super().__init__(subject_dir_path=self.subject_dir_path, **kwargs)
        self.trials = [ViconCsvTrial(folder) for folder in self.subject_dir_path.iterdir() if (folder.is_dir() and
                       folder.stem != 'Static')]

    @lazy
    def subject_df(self) -> pd.DataFrame:
        """A Pandas dataframe summarizing the Vicon CSV trials belonging to the subject."""
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Trial'] = pd.Series(self.trials, dtype=object)
        return df


class ViconStatic:
    """Marker positions from the static trial for a subject in the Vicon and biplane fluoroscopy database.

    Designed as a mix-in class.

    Attributes
    ----------
    static_trial_file: pathlib.Path
        Path to the file containing the Vicon to biplane fluoroscopy coordinate system transformation data.
    """

    def __init__(self, static_trial_file: Union[Callable, Path], **kwargs):
        if callable(static_trial_file):
            self.static_file = static_trial_file()
        else:
            self.static_file = static_trial_file
        assert(self.static_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def vicon_static_data(self) -> pd.DataFrame:
        """Pandas dataframe with the data from the static trial."""
        return pd.read_csv(self.static_file, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def static(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to static marker data. The indexed access
        returns a (n, 3) numpy array view."""
        return NestedDescriptor(self.vicon_static_data, csv_get_item_method_squeeze)


class BiplaneViconSubject(SubjectDescription, ViconCSTransform, ViconStatic):
    """A subject that contains multiple BiplaneVicon trials.

    Attributes
    ----------
    subject_dir_path: pathlib.Path
        Path to directory containing subject data.
    trials: list of biplane_kine.database.biplane_vicon_db.BiplaneViconTrial
        List of trials for the subject.
    """

    def __init__(self, subj_dir: Union[str, Path], **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        def f_t_v_file(): return self.subject_dir_path / 'Static' / (self.subject_name + '_F_T_V.csv')
        def static_file(): return self.subject_dir_path / 'Static' / (self.subject_name + '_vicon_static_trial.csv')
        super().__init__(subject_dir_path=self.subject_dir_path, f_t_v_file=f_t_v_file, static_trial_file=static_file,
                         **kwargs)
        self.trials = [BiplaneViconTrial(folder, self) for
                       folder in self.subject_dir_path.iterdir() if (folder.is_dir() and folder.stem != 'Static')]

    @lazy
    def subject_df(self) -> pd.DataFrame:
        """A Pandas dataframe summarizing the Vicon CSV trials belonging to the subject."""
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Trial'] = pd.Series(self.trials, dtype=object)
        df['Subject'] = pd.Series([self] * len(self.trials))
        return df
