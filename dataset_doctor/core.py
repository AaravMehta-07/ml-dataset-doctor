import pandas as pd
from .checks import run_all_checks, drift_check
from .report import Report
from .planner import build_fix_plan
from .fixes import apply_fix_plan

def scan(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    results = run_all_checks(df, target)
    fix_plan = build_fix_plan(results)
    return Report(results, fix_plan)

def apply_fixes(df: pd.DataFrame, fix_plan: dict):
    return apply_fix_plan(df, fix_plan)

def compare_drift(ref_df: pd.DataFrame, new_df: pd.DataFrame):
    return drift_check(ref_df, new_df)
