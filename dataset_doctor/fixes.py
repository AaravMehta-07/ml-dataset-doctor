import pandas as pd
import numpy as np

def apply_fix_plan(df: pd.DataFrame, plan: dict):
    df = df.copy()
    log = []

    if plan.get("drop_duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        log.append({"action": "drop_duplicates", "rows_removed": before - len(df)})

    if "drop_constant_features" in plan:
        cols = plan["drop_constant_features"]
        df = df.drop(columns=cols)
        log.append({"action": "drop_constant_features", "columns": cols})

    if plan.get("impute_missing") == "median":
        for c in df.select_dtypes(include=np.number):
            if df[c].isnull().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
                log.append({"action": "impute_missing", "column": c, "method": "median"})

    if plan.get("cap_outliers") == "iqr":
        for c in df.select_dtypes(include=np.number):
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            df[c] = df[c].clip(low, high)
            log.append({"action": "cap_outliers", "column": c, "method": "iqr"})

    return df, log
