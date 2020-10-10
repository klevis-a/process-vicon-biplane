"""This module provides access to Vicon C3D trials."""

from pathlib import Path
import numpy as np
import pandas as pd
from lazy import lazy
from ezc3d import c3d
from typing import Union, Callable, Type
from .db_common import TrialDescription, ViconEndpts, SubjectDescription, trial_descriptor_df
from ..misc.python_utils import NestedDescriptor


class C3DHelper:
    """A Vicon C3D trial accessor.
    
    Attributes
    ----------
    marker_names: list of str
        The names of the markers present in this capture.
    analog_frame_rate: float
        Sampling frequency for analog signals expressed in Hz.
    marker_frame_rate: float
        Marker data sampling frequency expressed in Hz.
    marker_data: numpy.ndarray
        Marker position data contained in a (4, m, n) numpy ndarray - where m is the number of markers and n is the
        number of timepoints.
    frames: numpy.ndarray
        Frame indices, zero-indexed.
    """

    def __init__(self, c3d_path: str):

        # Note that we extract the relevant parameters from c3d_obj and then it should be garbage collected -
        # this results in a significant savings in memory.
        c3d_obj = c3d(c3d_path)
        self.marker_names = c3d_obj['parameters']['POINT']['LABELS']['value']
        self._marker_map = dict(zip(self.marker_names, range(len(self.marker_names))))
        self.analog_frame_rate = c3d_obj['header']['analogs']['frame_rate']
        self.marker_frame_rate = c3d_obj['header']['points']['frame_rate']
        self.marker_data = c3d_obj['data']['points']
        self.frames = np.arange(self.marker_data.shape[2])

    def __getitem__(self, marker_name):
        """Marker data, (nx4) numpy array view, associated with marker_name."""
        return self.marker_data[:, self._marker_map[marker_name], :].T


def c3d_get_item_method(c3d_helper: C3DHelper, marker_name: str) -> np.ndarray:
    """Return the marker data, (n, 4) numpy array view, associated with marker_name."""
    return c3d_helper[marker_name]


class C3DTrial(TrialDescription):
    """A Vicon trial that has both labeled and filled C3D data.

    Enables lazy (and cached) access to the labeled andfilled Vicon Data.

    Attributes
    ----------
    labeled_c3d_path: pathlib.Path
        Path to c3d file that contains the labeled Vicon marker data.
    filled_c3d_path: pathlib.Path
        Path to c3d file that contains the filled Vicon marker data.
    """

    def __init__(self, labeled_c3d_path: Union[str, Path], filled_c3d_path: Union[str, Path], **kwargs):
        self.labeled_c3d_path = labeled_c3d_path if isinstance(labeled_c3d_path, Path) else Path(labeled_c3d_path)
        self.filled_c3d_path = filled_c3d_path if isinstance(filled_c3d_path, Path) else Path(filled_c3d_path)
        assert(self.labeled_c3d_path.is_file())
        assert(self.filled_c3d_path.is_file())

        super().__init__(trial_dir_path=self.labeled_c3d_path, **kwargs)

    @lazy
    def labeled_c3d(self) -> C3DHelper:
        """Labeled C3D data accessor."""
        return C3DHelper(str(self.labeled_c3d_path))

    @lazy
    def filled_c3d(self) -> C3DHelper:
        """Filled C3D data accessor."""
        return C3DHelper(str(self.filled_c3d_path))

    @lazy
    def labeled(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to labeled C3D data. The indexed access returns
         a (n, 4) numpy array view associated with marker_name."""
        return NestedDescriptor(self.labeled_c3d, c3d_get_item_method)

    @lazy
    def filled(self) -> NestedDescriptor:
        """Descriptor that allows marker indexed ([marker_name]) access to filled C3D data. The indexed access returns
        a (n, 4) numpy array view associated with marker_name."""
        return NestedDescriptor(self.filled_c3d, c3d_get_item_method)


class C3DTrialEndpts(C3DTrial, ViconEndpts):
    """A Vicon trial that has both labeled and filled C3D data, and also indicates which of its frame indices
    (endpoints) correspond to the endpoints of the reciprocal biplane fluoroscopy trial.

    Enables lazy (and cached) access to the labeled and filled Vicon Data.
    """

    def __init__(self, labeled_c3d_path: Union[str, Path], filled_c3d_path: Union[str, Path],
                 endpts_path: Union[Callable, Path]):
        super().__init__(labeled_c3d_path=labeled_c3d_path, filled_c3d_path=filled_c3d_path,
                         endpts_file=lambda: endpts_path / (self.trial_name + '_vicon_endpts.csv'))


class C3DSubjectEndpts(SubjectDescription):
    """A subject that contains multiple Vicon C3D trials.

    Attributes
    ----------
    subject_dir_path: pathlib.Path
        Path to directory where subject data resides.
    static_dir: pathlib.Path
        Path to directory where static (anything other than dynamic trial) subject data resides.
    labeled_base_path: pathlib.Path
        Path to directory where labeled Vicon marker data resides.
    filled_base_path: pathlib.Path
        Path to directory where filled Vicon marker data resides.
    trials: list of biplane_kine.database.c3d_helper.C3DTrial
        List of trials for the subject.
    """

    def __init__(self, subject_dir: Union[str, Path], labeled_base_dir: Union[str, Path],
                 filled_base_dir: Union[str, Path], c3d_trial_cls: Type[C3DTrial] = C3DTrialEndpts):
        self.subject_dir_path = subject_dir if isinstance(subject_dir, Path) else Path(subject_dir)
        self.static_dir = self.subject_dir_path / 'Static'
        super().__init__(subject_dir_path=self.subject_dir_path)

        self.labeled_base_path = labeled_base_dir if isinstance(labeled_base_dir, Path) else Path(labeled_base_dir)
        self.filled_base_path = filled_base_dir if isinstance(filled_base_dir, Path) else Path(filled_base_dir)

        # trials
        self.trials = [c3d_trial_cls(self.labeled_base_path / self.subject_name / (trial_dir.stem + '.c3d'),
                                     self.filled_base_path / self.subject_name / (trial_dir.stem + '.c3d'),
                                     endpts_path=trial_dir)
                       for trial_dir in self.subject_dir_path.iterdir()
                       if (trial_dir.is_dir() and trial_dir.name != 'Static')]

    @lazy
    def subject_df(self) -> pd.DataFrame:
        """A Pandas dataframe summarizing the Vicon C3D trials belonging to the subject."""
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Trial'] = pd.Series(self.trials, dtype=object)
        return df
