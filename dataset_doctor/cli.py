import argparse
import json
from pathlib import Path

import pandas as pd

from .core import apply_fixes, compare_drift, scan


def _load_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def main():
    p = argparse.ArgumentParser(description="Diagnose, fix, and audit ML-ready tabular datasets.")
    p.add_argument("csv", help="CSV file to scan")
    p.add_argument("--target", required=True, help="Target/label column")
    p.add_argument("--json", action="store_true", help="Print full JSON report")
    p.add_argument("--report", help="Write JSON report to this path")
    p.add_argument("--auto-fix", action="store_true", help="Apply safe automatic fixes")
    p.add_argument("--output", default="data_cleaned.csv", help="Output CSV path for --auto-fix")
    p.add_argument("--fix-log", default="fix_log.json", help="Fix log path for --auto-fix")
    p.add_argument("--compare", help="Optional new CSV to compare for drift against csv")
    args = p.parse_args()

    df = _load_csv(args.csv)
    report = scan(df, args.target)

    if args.compare:
        new_df = _load_csv(args.compare)
        drift = compare_drift(df, new_df)
        payload = report.to_dict()
        payload["drift"] = drift
        output_json = json.dumps(payload, indent=2, default=str)
    else:
        output_json = report.to_json()

    if args.report:
        Path(args.report).write_text(output_json, encoding="utf-8")

    if args.auto_fix and report.fix_plan:
        fixed_df, log = apply_fixes(df, report.fix_plan)
        fixed_df.to_csv(args.output, index=False)
        Path(args.fix_log).write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    print(output_json if args.json else report)


if __name__ == "__main__":
    main()