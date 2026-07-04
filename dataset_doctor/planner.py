def build_fix_plan(results):
    plan = {}

    if results.get("duplicates", 0) > 0:
        plan["drop_duplicates"] = True

    if results.get("constant_features"):
        plan["drop_constant_features"] = results["constant_features"]

    missing = {k: v for k, v in results.get("missing_values", {}).items() if v > 0}
    target = results.get("profile", {}).get("target")
    if missing:
        numeric = set(results.get("profile", {}).get("numeric_features", []))
        categorical = set(results.get("profile", {}).get("categorical_features", []))
        plan["impute_missing"] = {
            "numeric": sorted(c for c in missing if c in numeric),
            "categorical": sorted(c for c in missing if c in categorical),
        }
        if target in missing:
            plan["drop_missing_target"] = True

    outlier_cols = [k for k, v in results.get("outliers", {}).items() if v.get("count", 0) > 0]
    if outlier_cols:
        plan["cap_outliers"] = {"method": "iqr", "columns": outlier_cols}

    if results.get("id_like_features"):
        plan["review_id_like_features"] = results["id_like_features"]

    if results.get("high_cardinality_features"):
        plan["review_high_cardinality_features"] = sorted(results["high_cardinality_features"].keys())

    if results.get("leakage"):
        plan["review_leakage_features"] = sorted(results["leakage"].keys())

    if results.get("feature_correlations"):
        plan["review_correlated_features"] = sorted(results["feature_correlations"].keys())

    target_health = results.get("target_health", {})
    if target_health.get("majority_class_ratio", 0) >= 0.9 and target_health.get("unique", 0) > 1:
        plan["review_class_imbalance"] = True

    return plan