def build_fix_plan(results):
    plan = {}

    if results["duplicates"] > 0:
        plan["drop_duplicates"] = True

    if results["constant_features"]:
        plan["drop_constant_features"] = results["constant_features"]

    missing = {k: v for k, v in results["missing_values"].items() if v > 0}
    if missing:
        plan["impute_missing"] = "median"

    outliers = {k: v for k, v in results["outliers"].items() if v > 0}
    if outliers:
        plan["cap_outliers"] = "iqr"

    return plan
