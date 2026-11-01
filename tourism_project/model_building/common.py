"""Shared schema, deterministic cleaning and feature availability rules."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'tourism_project/data/tourism.csv'
SPLITS = ROOT / 'artifacts/splits'
DEPLOY = ROOT / 'tourism_project/deployment'
REPORTS = ROOT / 'reports'
TARGET = 'ProdTaken'
EXPECTED = ['CustomerID', TARGET, 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
# These fields describe an interaction which has already happened.
POST_CONTACT = ['TypeofContact', 'DurationOfPitch', 'NumberOfFollowups',
                'ProductPitched', 'PitchSatisfactionScore']
NUMERIC = ['Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
           'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']
CATEGORICAL = ['Occupation', 'Gender', 'MaritalStatus', 'Designation']
FEATURES = NUMERIC + CATEGORICAL

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str))

def clean_data(frame):
    """Only fixed rules here. Learned imputation lives inside cross-validation."""
    d = frame.drop(columns=[c for c in frame if c.startswith('Unnamed:')]).copy()
    for c in d.select_dtypes('object'):
        d[c] = d[c].str.strip().replace('', np.nan)
    d['Gender'] = d['Gender'].replace({'Fe Male': 'Female'})
    # Keep Single and Unmarried distinct: the source does not define them as synonyms.
    # IDs do not belong in a predictor; removing them reveals repeated records.
    d = d.drop(columns='CustomerID').drop_duplicates().reset_index(drop=True)
    return d

def profile_groups(x):
    """Same observed pre-contact profile must stay within one data partition."""
    return pd.util.hash_pandas_object(x[FEATURES], index=False).astype(str)
