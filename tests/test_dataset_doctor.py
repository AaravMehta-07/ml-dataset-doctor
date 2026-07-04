import json
import pandas as pd

from dataset_doctor import apply_fixes, compare_drift, scan


def test_scan_handles_categorical_target_and_builds_plan():
    df = pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d", "e", "f"],
            "x": [1, 2, 100, None, 2, 2],
            "cat": ["a", "a", None, "b", "a", "a"],
            "const": [1, 1, 1, 1, 1, 1],
            "y": ["yes", "no", "yes", "no", "no", "no"],
        }
    )
    report = scan(df, "y")

    assert report.results["target_health"]["is_numeric"] is False
    assert "const" in report.fix_plan["drop_constant_features"]
    assert "cat" in report.fix_plan["impute_missing"]["categorical"]
    assert report.to_dict()["risk_score"] > 0
    json.loads(report.to_json())


def test_apply_fixes_imputes_categorical_and_numeric_values():
    df = pd.DataFrame(
        {
            "x": [1, 2, 100, None, 2, 2],
            "cat": ["a", "a", None, "b", "a", "a"],
            "const": [1, 1, 1, 1, 1, 1],
            "y": ["yes", "no", "yes", "no", "no", "no"],
        }
    )
    report = scan(df, "y")
    fixed, log = apply_fixes(df, report.fix_plan)

    assert "const" not in fixed.columns
    assert fixed["x"].isna().sum() == 0
    assert fixed["cat"].isna().sum() == 0
    assert log


def test_drift_handles_mismatched_columns_and_categoricals():
    ref = pd.DataFrame({"x": [1, 2, 3, 4, 5], "cat": ["a", "a", "b", "b", "b"], "old": [1, 1, 1, 1, 1]})
    new = pd.DataFrame({"x": [5, 6, 7, 8, 9], "cat": ["a", "c", "c", "c", "c"], "new": [0, 0, 0, 0, 0]})

    drift = compare_drift(ref, new)

    assert drift["missing_columns"] == ["old"]
    assert drift["new_columns"] == ["new"]
    assert "x" in drift["numeric"]
    assert "cat" in drift["categorical"]