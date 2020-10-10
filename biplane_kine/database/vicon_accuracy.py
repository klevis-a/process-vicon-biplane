"""This module provides access to biplane fluoroscopy trials in which Vicon markers were tracked to ascertain the
spatiotemporal syncing accuracy between the Vicon and biplane fluoroscopy systems."""

from pathlib import Path
import numpy as np
import pandas as pd
from lazy import lazy
from typing import Union, Dict, NamedTuple, Iterator, Tuple
from .db_common import ViconEndpts, MARKERS, TrialDescription, SubjectDescription, ViconCSTransform, trial_descriptor_df


class ViconAccuracyMarkerData(NamedTuple):
    """A NamedTuple class that holds the data for a Vicon marker tracked using biplane fluoroscopy.

    Attributes
    ----------
    indices: numpy.ndarray
        Holds the frame indices for which the Vicon marker was tracked.
    data: numpy.ndarray
        Holds the tracked Vicon marker data expressed in the biplane fluoroscopy coordinate system.
    """
    indices: np.ndarray
    data: np.ndarray


class BiplaneMarkerTrial(TrialDescription):
    """A biplane fluorscopy trial in which Vicon markers were tracked to ascertain the spatiotemporal syncing accuracy
    between the Vicon and biplane fluoroscopy systems.

    Attributes
    ----------
    trial_dir_path: pathlib.Path
        Path to the directory where the trial data resides.
    subject: biplane_kine.database.vicon_accuracy.BiplaneMarkerSubjectEndpts
        Pointer to the subject that contains this trial.
    markers: dict of {str, ViconAccuracyMarkerData}
        Map from marker name to the data associated with that marker.
    """

    _VICON_ACCURACY_FILE_HEADERS = {'frame': np.int32, 'x': np.float64, 'y': np.float64, 'z': np.float64}

    def __init__(self, trial_dir: Union[str, Path], subject: 'BiplaneMarkerSubjectEndpts', **kwargs):
        self.trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        super().__init__(trial_dir_path=self.trial_dir_path, **kwargs)

        self.subject = subject
        self._marker_files = {file.stem.split('_')[-1]: file for file in self.trial_dir_path.iterdir() if
                              (file.stem.split('_')[-1] in MARKERS)}
        self.markers = self._marker_files.keys()

    @staticmethod
    def _read_marker_file(file_path: Union[str, Path]) -> pd.DataFrame:
        return pd.read_csv(file_path, header=0, dtype=BiplaneMarkerTrial._VICON_ACCURACY_FILE_HEADERS,
                           index_col='frame')

    @staticmethod
    def _process_marker_file(file_path: Union[str, Path]) -> ViconAccuracyMarkerData:
        marker_data = BiplaneMarkerTrial._read_marker_file(file_path)
        frames = marker_data.index.to_numpy()
        return ViconAccuracyMarkerData(frames - 1, marker_data.to_numpy())

    @lazy
    def _marker_data(self) -> Dict[str, ViconAccuracyMarkerData]:
        return {k: BiplaneMarkerTrial._process_marker_file(v) for k, v in self._marker_files.items()}

    def __getitem__(self, marker_name: str) -> ViconAccuracyMarkerData:
        """Marker based indexing ([marker_name]) to the Vicon markers as tracked via biplane fluoroscopy.

        Returns a namedtuple providing access to the frame indices (indices, (n,) numpy array) and marker positions
        (data, (n,3) numpy array view) in the biplane fluoroscopy CS."""
        return self._marker_data[marker_name]

    def __iter__(self) -> Iterator[Tuple[str, ViconAccuracyMarkerData]]:
        """Iterator over the markers that were tracked via biplane fluoroscopy and their associated data."""
        return (i for i in self._marker_data.items())


class BiplaneMarkerTrialEndpts(BiplaneMarkerTrial, ViconEndpts):
    """A biplane fluorscopy trial with tracked Vicon markers that also indicates the endpoints (frame indices) of the
    reciprocal Vicon trial that correspond to the endpoints of this trial."""

    def __init__(self, trial_dir: Union[str, Path], subject: 'BiplaneMarkerSubjectEndpts', **kwargs):
        trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        super().__init__(trial_dir=trial_dir_path, subject=subject,
                         endpts_file=lambda: trial_dir_path / (self.trial_name + '_vicon_endpts.csv'), **kwargs)


class BiplaneMarkerSubjectEndpts(SubjectDescription, ViconCSTransform):
    """A subject that contains multiple biplane fluoroscopy trials with tracked Vicon markers.
    
    Attributes
    ----------
    subject_dir_path: pathlib.Path
        Path to directory containing subject data.
    trials: list of biplane_kine.database.vicon_accuracy.BiplaneMarkerTrialEndpts
        List of trials for the subject.
    """

    def __init__(self, subj_dir: Union[str, Path], **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        super().__init__(subject_dir_path=self.subject_dir_path,
                         f_t_v_file=lambda: self.subject_dir_path / (self.subject_name + '_F_T_V.csv'), **kwargs)
        self.trials = [BiplaneMarkerTrialEndpts(folder, self) for folder in self.subject_dir_path.iterdir()
                       if folder.is_dir()]

    @lazy
    def subject_df(self) -> pd.DataFrame:
        """A Pandas dataframe summarizing the biplane fluoroscopy trials belonging to the subject."""
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Biplane_Marker_Trial'] = pd.Series(self.trials, dtype=object)
        return df
