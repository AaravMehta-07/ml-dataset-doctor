import numpy as np
import pandas as pd


def _mode_or_unknown(series):
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return "UNKNOWN"


def apply_fix_plan(df: pd.DataFrame, plan: dict):
    df = df.copy()
    log = []

    if plan.get("drop_missing_target"):
        before = len(df)
        target = plan.get("target")
        if target and target in df.columns:
            df = df[df[target].notna()]
            log.append({"action": "drop_missing_target", "rows_removed": before - len(df), "target": target})

    if plan.get("drop_duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        log.append({"action": "drop_duplicates", "rows_removed": before - len(df)})

    cols = [c for c in plan.get("drop_constant_features", []) if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
        log.append({"action": "drop_constant_features", "columns": cols})

    impute = plan.get("impute_missing")
    if impute == "median":
        impute = {"numeric": df.select_dtypes(include=np.number).columns.tolist(), "categorical": []}

    if isinstance(impute, dict):
        for c in impute.get("numeric", []):
            if c in df.columns and df[c].isnull().any():
                med = df[c].median()
                if pd.isna(med):
                    med = 0
                df[c] = df[c].fillna(med)
                log.append({"action": "impute_missing", "column": c, "method": "median", "value": float(med)})
        for c in impute.get("categorical", []):
            if c in df.columns and df[c].isnull().any():
                value = _mode_or_unknown(df[c])
                df[c] = df[c].fillna(value)
                log.append({"action": "impute_missing", "column": c, "method": "mode", "value": str(value)})

    cap = plan.get("cap_outliers")
    if cap == "iqr":
        columns = df.select_dtypes(include=np.number).columns.tolist()
    elif isinstance(cap, dict) and cap.get("method") == "iqr":
        columns = cap.get("columns", [])
    else:
        columns = []

    for c in columns:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = df[c].copy()
        df[c] = df[c].clip(low, high)
        changed = int((before != df[c]).sum())
        if changed:
            log.append({"action": "cap_outliers", "column": c, "method": "iqr", "changed": changed, "low": float(low), "high": float(high)})

    return df, log