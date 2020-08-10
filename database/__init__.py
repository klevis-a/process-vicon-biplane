from pathlib import Path
import pandas as pd
import numpy as np
from .db_subject import Subject


def create_db(db_dir):
    db_path = Path(db_dir)
    subjects = [Subject(subject_dir) for subject_dir in db_path.iterdir() if subject_dir.is_dir()]
    subject_dfs = [Subject.create_subject_df(subject) for subject in subjects]
    db = pd.concat(subject_dfs, ignore_index=True)
    db[['Subject', 'Trial_Name', 'Subject_Short']] = db[['Subject', 'Trial_Name', 'Subject_Short']].astype('string')
    db.set_index('Trial_Name', drop=True, inplace=True, verify_integrity=True)
    db.attrs['dt'] = 1/100

    anthro_file = db_path / 'Subject_Anthropometrics.csv'
    subject_anthro_dtype = {'Subject': 'string', 'Dominant_Arm': 'string', 'Gender': 'string', 'Age': np.int32,
                            'Height': np.float64,
                            'Weight': np.float64, 'BMI': np.float64, 'Armpit_Thickness': np.float64,
                            'Hand_Thickness': np.float64}
    subject_anthro = pd.read_csv(anthro_file, header=0, dtype=subject_anthro_dtype, index_col='Subject')
    return db, subject_anthro


def pre_initialize(db):
    for trial in db['Trial']:
        trial.humerus_biplane_data
        trial.scapula_biplane_data
        trial.vicon_data
        trial.vicon_endpts
