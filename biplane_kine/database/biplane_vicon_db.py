"""This module provides access to the Vicon and biplane fluoroscopy filesystem-based database."""

from pathlib import Path
import itertools
import functools
import numpy as np
import pandas as pd
from lazy import lazy
from typing import Union, Callable, Type
from scipy.spatial.transform import Rotation
from ..kinematics.joint_cs import torso_cs_isb, torso_cs_v3d
from ..kinematics.segments import StaticTorsoSegment
from .db_common import TrialDescription, ViconEndpts, SubjectDescription, ViconCSTransform, trial_descriptor_df, MARKERS
from ..misc.python_utils import NestedDescriptor

BIPLANE_FILE_HEADERS = {'frame': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                        'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}
TORSO_FILE_HEADERS = {'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                      'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}
TORSO_TRACKING_MARKERS = ['STRN', 'C7', 'T5', 'T10', 'CLAV']


def csv_get_item_method(csv_data: pd.DataFrame, marker_name: str) -> np.ndarray:
    """Return the marker data, (n, 3) numpy array view, associated with marker_name."""
    return csv_data.loc[:, marker_name:(marker_name + '.2')].to_numpy()


def csv_get_item_method_squeeze(csv_data: pd.DataFrame, marker_name: str) -> np.ndarray:
    """Return the marker data, (n, 3) numpy array view, associated with marker_name."""
    return np.squeeze(csv_get_item_method(csv_data, marker_name))


def insert_nans(func: Callable) -> Callable:
    """Return a new dataframe derived from the original dataframe with appended columns filled with NaNs for missing
    markers."""
    @functools.wraps(func)
    def wrapper(self) -> pd.DataFrame:
        orig_data = func(self)
        if not self.nan_missing_markers:
            return orig_data

        new_columns = [marker for marker in MARKERS if marker not in orig_data.columns]
        new_columns1 = [col + '.1' for col in new_columns]
        new_columns2 = [col + '.2' for col in new_columns]
        raw_data = orig_data.to_numpy()
        data_with_nan = np.concatenate((raw_data, np.full((orig_data.shape[0], len(new_columns) * 3), np.nan)), 1)
        all_columns = itertools.chain(orig_data.columns,
                                      itertools.chain.from_iterable(zip(new_columns, new_columns1, new_columns2)))
        return pd.DataFrame(data=data_with_nan, columns=all_columns, dtype=np.float64)
    return wrapper


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
    nan_missing_markers: bool
        Specifies whether to insert NaNs in the dataset for missing markers
    """

    def __init__(self, trial_dir: Union[str, Path], nan_missing_markers: bool = False, **kwargs):
        self.nan_missing_markers = nan_missing_markers
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
    @insert_nans
    def vicon_csv_data_labeled(self) -> pd.DataFrame:
        """Pandas dataframe with the labeled Vicon CSV data."""
        # TODO: this works fine for now and by using the accessor method below we get a view (rather than a copy) of the
        #  data, however it probably makes sense to using something like structured arrays or xarray. Note that
        #  multi-level column labels should not be used (i.e. header=[0, 1) because a copy of the data, not a view is
        #  returned
        return pd.read_csv(self.vicon_csv_file_labeled, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    @insert_nans
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


class BiplaneViconTrial(ViconCsvTrial):
    """A trial that contains both biplane and Vicon data.

    Attributes
    ----------
    vicon_csv_file_smoothed: pathlib.Path
        Path to the smoothed marker data for the Vicon CSV trial.
    subject: biplane_kine.database.vicon_accuracy.BiplaneViconSubject
        Pointer to the subject that contains this trial.
    """

    def __init__(self, trial_dir: Union[str, Path], subject: 'BiplaneViconSubject', nan_missing_markers: bool = True,
                 **kwargs):
        super().__init__(trial_dir, nan_missing_markers, **kwargs)
        self.subject = subject
        # file paths
        self.vicon_csv_file_smoothed = self.trial_dir_path / (self.trial_name + '_vicon_smoothed.csv')
        self.humerus_biplane_file = self.trial_dir_path / (self.trial_name + '_humerus_biplane.csv')
        self.scapula_biplane_file = self.trial_dir_path / (self.trial_name + '_scapula_biplane.csv')
        self.torso_vicon_file = self.trial_dir_path / (self.trial_name + '_torso.csv')

        # make sure the files are actually there
        assert (self.vicon_csv_file_smoothed.is_file())
        assert (self.humerus_biplane_file.is_file())
        assert (self.scapula_biplane_file.is_file())
        assert (self.torso_vicon_file.is_file())

    @lazy
    @insert_nans
    def vicon_csv_data_smoothed(self) -> pd.DataFrame:
        """Pandas dataframe with the smoothed Vicon CSV data."""
        return pd.read_csv(self.vicon_csv_file_smoothed, header=[0], skiprows=[1], dtype=np.float64)

    @lazy
    def smoothed(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to smoothed CSV data. The indexed access returns
        a (n, 3) numpy array view."""
        return NestedDescriptor(self.vicon_csv_data_smoothed, csv_get_item_method)

    @lazy
    def humerus_biplane_data(self) -> pd.DataFrame:
        """Humerus raw biplane data."""
        return pd.read_csv(self.humerus_biplane_file, header=0, dtype=BIPLANE_FILE_HEADERS, index_col='frame')

    @lazy
    def scapula_biplane_data(self) -> pd.DataFrame:
        """Scapula raw biplane data."""
        return pd.read_csv(self.scapula_biplane_file, header=0, dtype=BIPLANE_FILE_HEADERS, index_col='frame')

    @staticmethod
    def homogeneous_from_biplane(df):
        quat_raw = df.iloc[:, 3:].to_numpy()
        # make quaternion scalar last
        quat_raw = np.concatenate((quat_raw[:, 1:], quat_raw[:, 0][..., np.newaxis]), 1)
        pos = df.iloc[:, :3].to_numpy()
        valid_idx = ~np.any(np.isnan(quat_raw), 1)
        rot_mat_valid = Rotation.from_quat(quat_raw[valid_idx]).as_matrix()
        ht_mat = np.full((quat_raw.shape[0], 4, 4), np.nan)
        ht_mat[:, :3, 3] = pos
        ht_mat[valid_idx, :3, :3] = rot_mat_valid
        ht_mat[:, 3, :] = [0, 0, 0, 1]
        return ht_mat

    @lazy
    def humerus_fluoro(self) -> np.ndarray:
        """Humerus frame trajectory (N, 4, 4) in biplane fluoroscopy reference frame."""
        return BiplaneViconTrial.homogeneous_from_biplane(self.humerus_biplane_data)

    @lazy
    def scapula_fluoro(self) -> np.ndarray:
        """Scapula frame trajectory (N, 4, 4) in biplane fluoroscopy reference frame."""
        return BiplaneViconTrial.homogeneous_from_biplane(self.scapula_biplane_data)

    @lazy
    def torso_vicon_data(self) -> pd.DataFrame:
        """Torso trajectory dataframe."""
        return pd.read_csv(self.torso_vicon_file, header=0, dtype=TORSO_FILE_HEADERS)

    @lazy
    def torso_vicon(self) -> np.ndarray:
        """Torso frame trajectory (N, 4, 4) in Vicon reference frame."""
        return BiplaneViconTrial.homogeneous_from_biplane(self.torso_vicon_data)


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

    def __init__(self, subj_dir: Union[str, Path], trial_class: Type[BiplaneViconTrial] = BiplaneViconTrial, **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        def f_t_v_file(): return self.subject_dir_path / 'Static' / (self.subject_name + '_F_T_V.csv')
        def static_file(): return self.subject_dir_path / 'Static' / (self.subject_name + '_vicon_static_trial.csv')
        super().__init__(subject_dir_path=self.subject_dir_path, f_t_v_file=f_t_v_file, static_trial_file=static_file,
                         **kwargs)
        # landmarks files
        self.humerus_landmarks_file = self.subject_dir_path / 'Static' / (self.subject_name + '_humerus_landmarks.csv')
        self.scapula_landmarks_file = self.subject_dir_path / 'Static' / (self.subject_name + '_scapula_landmarks.csv')
        assert(self.humerus_landmarks_file.is_file())
        assert(self.scapula_landmarks_file.is_file())

        self.trials = [trial_class(folder, self) for
                       folder in self.subject_dir_path.iterdir() if (folder.is_dir() and folder.stem != 'Static')]

    @lazy
    def subject_df(self) -> pd.DataFrame:
        """A Pandas dataframe summarizing the Vicon CSV trials belonging to the subject."""
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Trial'] = pd.Series(self.trials, dtype=object)
        df['Subject'] = pd.Series([self] * len(self.trials))
        return df

    @lazy
    def torso(self) -> StaticTorsoSegment:
        """Torso kinematic trajectory, numpy.ndarray (N, 4, 4)."""
        return StaticTorsoSegment(torso_cs_isb, self.static)


class BiplaneViconSubjectV3D(BiplaneViconSubject):
    """A subject that contains multiple BiplaneVicon trials.

    Creates torso coordinate system using the Visual3D definition.

    armpit_thickness: float
        Subject's armpit thickness, used to define a torso coordinate system according to Visual3D
    """

    def __init__(self, subj_dir: Union[str, Path], armpit_thickness,
                 trial_class: Type[BiplaneViconTrial] = BiplaneViconTrial, **kwargs):
        super().__init__(subj_dir, trial_class, **kwargs)
        self.armpit_thickness = armpit_thickness(self.subject_name)

    @lazy
    def torso(self) -> StaticTorsoSegment:
        torso_cs_v3d_func = functools.update_wrapper(
            functools.partial(torso_cs_v3d, armpit_thickness=self.armpit_thickness), torso_cs_v3d)
        return StaticTorsoSegment(torso_cs_v3d_func, self.static)
