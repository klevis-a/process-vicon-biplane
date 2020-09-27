from pathlib import Path
import numpy as np
import pandas as pd
from ezc3d import c3d
from .db_common import TrialDescriptor, ViconEndpts, SubjectDescriptor, trial_descriptor_df


class C3DHelper:
    def __init__(self, c3d_path):

        # Note that we extract the relevant parameters from c3d_obj and then it should be garbage collected -
        # this results in a significant savings in memory.
        c3d_obj = c3d(c3d_path)
        self.marker_names = c3d_obj['parameters']['POINT']['LABELS']['value']
        self.marker_map = dict(zip(self.marker_names, range(len(self.marker_names))))
        self.analog_frame_rate = c3d_obj['header']['analogs']['frame_rate']
        self.marker_frame_rate = c3d_obj['header']['points']['frame_rate']
        self.marker_data = c3d_obj['data']['points']
        self.frames = np.arange(self.marker_data.shape[2])

    def data_for_marker(self, marker_name):
        return self.marker_data[0:3, self.marker_map[marker_name], :].T


class C3DTrial(TrialDescriptor):
    def __init__(self, labeled_c3d_path, filled_c3d_path, **kwargs):
        self.labeled_c3d_path = labeled_c3d_path if isinstance(labeled_c3d_path, Path) else Path(labeled_c3d_path)
        self.filled_c3d_path = filled_c3d_path if isinstance(filled_c3d_path, Path) else Path(filled_c3d_path)
        assert(self.labeled_c3d_path.is_file())
        assert(self.filled_c3d_path.is_file())

        super().__init__(trial_dir_path=self.labeled_c3d_path, **kwargs)

        self._labeled_c3d = None
        self._filled_c3d = None

    @property
    def labeled_c3d(self):
        if self._labeled_c3d is None:
            self._labeled_c3d = C3DHelper(str(self.labeled_c3d_path))
        return self._labeled_c3d

    @property
    def filled_c3d(self):
        if self._filled_c3d is None:
            self._filled_c3d = C3DHelper(str(self.filled_c3d_path))
        return self._filled_c3d

    def marker_data_labeled(self, marker_name):
        return self.labeled_c3d.data_for_marker(marker_name)

    def marker_data_filled(self, marker_name):
        return self.filled_c3d.data_for_marker(marker_name)


class C3DTrialEndpts(C3DTrial, ViconEndpts):
    def __init__(self, labeled_c3d_path, filled_c3d_path, endpts_file):
        super().__init__(labeled_c3d_path=labeled_c3d_path, filled_c3d_path=filled_c3d_path,
                         endpts_file=Path(endpts_file))


class C3DSubjectEndpts(SubjectDescriptor):
    def __init__(self, subject_dir, labeled_base_dir, filled_base_dir):
        self.subject_dir_path = subject_dir if isinstance(subject_dir, Path) else Path(subject_dir)
        self.static_dir = self.subject_dir_path / 'Static'
        super().__init__(subject_dir_path=self.subject_dir_path)

        self.labeled_base_path = labeled_base_dir if isinstance(labeled_base_dir, Path) else Path(labeled_base_dir)
        self.filled_base_path = filled_base_dir if isinstance(filled_base_dir, Path) else Path(filled_base_dir)

        # trials
        self.trials = [C3DTrialEndpts(self.labeled_base_path / self.subject_name / (trial_dir.stem + '.c3d'),
                                      self.filled_base_path / self.subject_name / (trial_dir.stem + '.c3d'),
                                      trial_dir / 'vicon_endpts.csv')
                       for trial_dir in self.subject_dir_path.iterdir()
                       if (trial_dir.is_dir() and trial_dir.name != 'Static')]

        # used for dataframe
        self._df = None

    @property
    def subject_df(self):
        if self._df is None:
            self._df = trial_descriptor_df(self.subject_name, self.trials)
            self._df['Trial'] = pd.Series(self.trials, dtype=object)
        return self._df
