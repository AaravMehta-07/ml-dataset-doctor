import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def missing_values(df):
    return df.isnull().mean().to_dict()

def duplicates(df):
    return int(df.duplicated().sum())

def constant_features(df):
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

def outliers(df):
    out = {}
    for c in df.select_dtypes(include=np.number):
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        out[c] = int(((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum())
    return out

def label_noise(df, target):
    X = df.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
    y = df[target]
    if X.empty or y.nunique() < 2:
        return None
    model = KNeighborsClassifier(5)
    model.fit(X, y)
    preds = model.predict(X)
    return float((preds != y).mean())

def leakage(df, target):
    corr = df.select_dtypes(include=np.number).corr()[target].abs()
    return corr[corr > 0.95].drop(target, errors="ignore").to_dict()

def run_all_checks(df, target):
    return {
        "shape": df.shape,
        "missing_values": missing_values(df),
        "duplicates": duplicates(df),
        "constant_features": constant_features(df),
        "outliers": outliers(df),
        "label_noise": label_noise(df, target),
        "leakage": leakage(df, target)
    }

def drift_check(ref_df, new_df):
    drift = {}
    for c in ref_df.select_dtypes(include=np.number):
        drift[c] = float(abs(ref_df[c].mean() - new_df[c].mean()))
    return drift
