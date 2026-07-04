import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, tuple):
        return list(value)
    return value


def dataset_profile(df, target=None):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 4),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target": target,
    }


def schema(df):
    return {
        col: {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "missing": int(df[col].isna().sum()),
        }
        for col in df.columns
    }


def missing_values(df):
    return {c: float(v) for c, v in df.isnull().mean().items()}


def missing_summary(df):
    return {
        c: {"count": int(df[c].isna().sum()), "ratio": float(df[c].isna().mean())}
        for c in df.columns
        if df[c].isna().any()
    }


def duplicates(df):
    return int(df.duplicated().sum())


def duplicate_columns(df):
    return df.columns[df.columns.duplicated()].tolist()


def constant_features(df, target=None):
    return [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]


def high_cardinality_features(df, target=None, threshold=0.5):
    result = {}
    for c in df.select_dtypes(exclude=np.number).columns:
        if c == target:
            continue
        ratio = float(df[c].nunique(dropna=True) / max(len(df), 1))
        if ratio >= threshold and len(df) >= 20:
            result[c] = {"unique": int(df[c].nunique(dropna=True)), "ratio": round(ratio, 4)}
    return result


def id_like_features(df, target=None):
    candidates = []
    for c in df.columns:
        if c == target or len(df) < 5:
            continue
        unique_ratio = df[c].nunique(dropna=True) / max(df[c].notna().sum(), 1)
        name_hint = any(token in c.lower() for token in ("id", "uuid", "guid", "hash"))
        if unique_ratio > 0.98 and (name_hint or df[c].dtype == object):
            candidates.append(c)
    return candidates


def outliers(df, target=None):
    out = {}
    for c in df.select_dtypes(include=np.number):
        if c == target:
            continue
        series = df[c].dropna()
        if series.empty:
            out[c] = {"count": 0, "ratio": 0.0, "low": None, "high": None}
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            out[c] = {"count": 0, "ratio": 0.0, "low": float(q1), "high": float(q3)}
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < low) | (series > high)).sum())
        out[c] = {"count": count, "ratio": float(count / max(len(series), 1)), "low": float(low), "high": float(high)}
    return out


def target_health(df, target):
    y = df[target]
    counts = y.value_counts(dropna=False)
    majority_ratio = float(counts.iloc[0] / len(y)) if len(y) else 0.0
    return {
        "dtype": str(y.dtype),
        "missing": int(y.isna().sum()),
        "unique": int(y.nunique(dropna=True)),
        "majority_class_ratio": round(majority_ratio, 4),
        "class_counts": {str(k): int(v) for k, v in counts.head(50).items()},
        "is_numeric": bool(pd.api.types.is_numeric_dtype(y)),
    }


def label_noise(df, target):
    X = df.drop(columns=[target]).select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target]
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if X.empty or len(y) < 6 or y.nunique() < 2:
        return None
    min_class_count = int(y.value_counts().min())
    n_neighbors = min(5, len(y), min_class_count)
    if n_neighbors < 1:
        return None
    try:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X, y)
        preds = model.predict(X)
        return float((preds != y).mean())
    except Exception:
        return None


def _encode_target(y):
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(float), True
    encoded = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    return encoded, False


def leakage(df, target):
    y_raw = df[target]
    valid = y_raw.notna()
    if valid.sum() < 3 or y_raw[valid].nunique() < 2:
        return {}
    y, is_regression = _encode_target(y_raw[valid])
    result = {}
    numeric = df.drop(columns=[target]).select_dtypes(include=np.number)
    for c in numeric.columns:
        x = numeric.loc[valid, c].replace([np.inf, -np.inf], np.nan)
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
            continue
        corr = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if pd.notna(corr) and corr > 0.95:
            result[c] = {"method": "correlation", "score": round(float(corr), 4)}

    features = numeric.loc[valid].replace([np.inf, -np.inf], np.nan).fillna(0)
    variable_features = [c for c in features.columns if features[c].nunique(dropna=True) > 1]
    features = features[variable_features]
    if not features.empty and len(features) >= 5:
        try:
            scores = (
                mutual_info_regression(features, y, random_state=42)
                if is_regression
                else mutual_info_classif(features, y, random_state=42)
            )
            for c, score in zip(features.columns, scores):
                if score >= 0.95 and c not in result:
                    result[c] = {"method": "mutual_info", "score": round(float(score), 4)}
        except Exception:
            pass
    return result


def feature_correlations(df, target=None, threshold=0.95):
    numeric = df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore")
    if numeric.shape[1] < 2:
        return {}
    corr = numeric.corr().abs()
    pairs = {}
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                pairs[f"{left}__{right}"] = round(float(value), 4)
    return pairs


def run_all_checks(df, target):
    return {
        "shape": tuple(df.shape),
        "profile": dataset_profile(df, target),
        "schema": schema(df),
        "missing_values": missing_values(df),
        "missing_summary": missing_summary(df),
        "duplicates": duplicates(df),
        "duplicate_columns": duplicate_columns(df),
        "constant_features": constant_features(df, target),
        "high_cardinality_features": high_cardinality_features(df, target),
        "id_like_features": id_like_features(df, target),
        "outliers": outliers(df, target),
        "label_noise": label_noise(df, target),
        "target_health": target_health(df, target),
        "leakage": leakage(df, target),
        "feature_correlations": feature_correlations(df, target),
    }


def _psi(ref, new, bins=10):
    ref = ref.dropna()
    new = new.dropna()
    if ref.empty or new.empty:
        return None
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return None
    ref_counts, _ = np.histogram(ref, bins=quantiles)
    new_counts, _ = np.histogram(new, bins=quantiles)
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), 1e-6)
    new_pct = np.maximum(new_counts / max(new_counts.sum(), 1), 1e-6)
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))


def drift_check(ref_df, new_df):
    drift = {"numeric": {}, "categorical": {}, "missing_columns": [], "new_columns": []}
    ref_cols, new_cols = set(ref_df.columns), set(new_df.columns)
    drift["missing_columns"] = sorted(ref_cols - new_cols)
    drift["new_columns"] = sorted(new_cols - ref_cols)
    for c in sorted(ref_cols & new_cols):
        if pd.api.types.is_numeric_dtype(ref_df[c]) and pd.api.types.is_numeric_dtype(new_df[c]):
            ref, new = ref_df[c].dropna(), new_df[c].dropna()
            mean_shift = float(abs(ref.mean() - new.mean())) if len(ref) and len(new) else None
            std = float(ref.std()) if len(ref) > 1 else 0.0
            normalized = float(mean_shift / std) if mean_shift is not None and std > 0 else None
            psi = _psi(ref, new)
            drift["numeric"][c] = {
                "mean_shift": _json_safe(mean_shift),
                "normalized_mean_shift": _json_safe(normalized),
                "psi": _json_safe(psi),
            }
        else:
            ref_dist = ref_df[c].astype(str).value_counts(normalize=True)
            new_dist = new_df[c].astype(str).value_counts(normalize=True)
            keys = set(ref_dist.index) | set(new_dist.index)
            total_variation = 0.5 * sum(abs(float(ref_dist.get(k, 0)) - float(new_dist.get(k, 0))) for k in keys)
            drift["categorical"][c] = {"total_variation": round(total_variation, 4)}
    return drift