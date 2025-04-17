from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, zscore
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def encode_categorical(df: pd.DataFrame, categorical_cols: list):
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            print(f"Column '{col}' not found.")
    return df, encoders

def remove_outliers(df: pd.DataFrame, numerical_cols: list, z_thresh: float = 2) -> pd.DataFrame:
    z_scores = df[numerical_cols].apply(zscore, nan_policy='omit')
    outlier_idx = np.where(np.abs(z_scores) > z_thresh)[0]
    return df.drop(outlier_idx)

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)
