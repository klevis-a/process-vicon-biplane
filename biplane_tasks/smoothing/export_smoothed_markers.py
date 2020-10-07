import numpy as np
import itertools
from biplane_kine.database.db_common import MARKERS
from biplane_kine.smoothing.kf_filtering_helpers import (piecewise_filter_with_exception, InsufficientDataError,
                                                         DoNotUseMarkerError)
import logging
from ..parameters import marker_smoothing_exceptions

log = logging.getLogger(__name__)


def export_to_csv(file_name, export_data, markers):
    header_line_1 = ','.join(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in markers))
    header_line_2 = ','.join(['x', 'y', 'z'] * len(markers))
    np.savetxt(file_name, np.concatenate(export_data, axis=1), delimiter=',', fmt='%.5g', comments='',
               header=header_line_1 + '\n' + header_line_2)


def trial_exporter(trial, dt, subj_dir, all_smoothing_except):
    log.info('Exporting trial %s', trial.trial_name)
    trial_dir = subj_dir / trial.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    data_to_export = []
    marker_names = []
    for marker in MARKERS:
        if marker in trial.vicon_csv_data_labeled.columns:
            log.info('Smoothing marker %s', marker)
            try:
                marker_exceptions = marker_smoothing_exceptions(all_smoothing_except, trial.trial_name, marker)
                raw, _, _, smoothed = \
                    piecewise_filter_with_exception(marker_exceptions, trial.labeled[marker], trial.filled[marker],
                                                    dt)
            except InsufficientDataError as e:
                log.warning('Skipping marker {} for trial {} because: {}'.format(marker, trial.trial_name, e))
                continue
            except DoNotUseMarkerError as e:
                log.warning('Skipping marker {} for trial {} because: {}'.format(marker, trial.trial_name, e))
                continue

            marker_export_data = np.full((raw.endpts[1], 3), np.nan)
            marker_export_data[smoothed.endpts[0]:smoothed.endpts[1], :] = smoothed.means.pos
            data_to_export.append(marker_export_data)
            marker_names.append(marker)
        else:
            log.warning('Marker %s missing', marker)
    export_to_csv(trial_dir / (trial.trial_name + '_vicon_smoothed.csv'), data_to_export, marker_names)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    from pathlib import Path
    from biplane_kine.database import create_db
    from biplane_kine.database.biplane_vicon_db import ViconCsvSubject
    from biplane_kine.misc.json_utils import Params
    from ..parameters import read_smoothing_exceptions
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    root_path = Path(params.output_dir)
    db = create_db(params.biplane_vicon_db_dir, ViconCsvSubject)
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)

    # create plots
    for subject_name, subject_df in db.groupby('Subject_Name'):
        log.info('Smoothing subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for t in subject_df['Trial']:
            trial_exporter(t, db.attrs['dt'], subject_dir, all_exceptions)
