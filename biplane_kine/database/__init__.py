from pathlib import Path
import pandas as pd
import numpy as np

subject_anthro_dtype = {'Subject': 'string', 'Dominant_Arm': 'string', 'Gender': 'string', 'Age': np.int32,
                        'Height': np.float64,
                        'Weight': np.float64, 'BMI': np.float64, 'Armpit_Thickness': np.float64,
                        'Hand_Thickness': np.float64}


def create_db(db_dir, subject_class):
    db_path = Path(db_dir)
    subjects = [subject_class(subject_dir) for subject_dir in db_path.iterdir() if subject_dir.is_dir()]
    subject_dfs = [subject.subject_df for subject in subjects]
    db = pd.concat(subject_dfs, ignore_index=True)
    db[['Subject', 'Trial_Name', 'Subject_Short']] = db[['Subject', 'Trial_Name', 'Subject_Short']].astype('string')
    db.set_index('Trial_Name', drop=False, inplace=True, verify_integrity=True)
    db.attrs['dt'] = 1/100
    return db


def anthro_db(db_dir):
    db_path = Path(db_dir)
    anthro_file = db_path / 'Subject_Anthropometrics.csv'
    subject_anthro = pd.read_csv(anthro_file, header=0, dtype=subject_anthro_dtype, index_col='Subject')
    return subject_anthro
