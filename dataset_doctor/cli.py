import argparse, pandas as pd, json
from .core import scan, apply_fixes

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--target", required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--auto-fix", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    report = scan(df, args.target)

    if args.auto_fix and report.fix_plan:
        df, log = apply_fixes(df, report.fix_plan)
        df.to_csv("data_cleaned.csv", index=False)
        with open("fix_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print(report.to_json() if args.json else report)

if __name__ == "__main__":
    main()
