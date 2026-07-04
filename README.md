<p align="center">
  <img src="https://img.shields.io/badge/ml--dataset--doctor-v0.3.0-6C63FF?style=for-the-badge&logo=python&logoColor=white" alt="version"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="license"/>
  <img src="https://img.shields.io/badge/scikit--learn-powered-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/PRs-welcome-ec4899?style=for-the-badge" alt="prs"/>
</p>

<h1 align="center">🩺 ml-dataset-doctor</h1>

<p align="center">
  <b>The definitive pre-training audit tool for tabular ML datasets.</b><br/>
  Scan. Diagnose. Fix. Ship clean data — in seconds.
</p>

<p align="center">
  <a href="#-features">Features</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-cli-usage">CLI</a> ·
  <a href="#-python-api">Python API</a> ·
  <a href="#-what-gets-detected">What Gets Detected</a> ·
  <a href="#-auto-fix-engine">Auto-Fix</a> ·
  <a href="#-drift-detection">Drift Detection</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-contributing">Contributing</a>
</p>

---

## 🚀 Why ml-dataset-doctor?

You wouldn't train a model on mystery data — yet most teams ship straight from raw CSVs to their pipeline with zero auditing. `ml-dataset-doctor` is the **last line of defense** before your data hits your model:

- 🔍 **15+ automated checks** in a single command
- 🤖 **Auto-fix engine** that safely patches common issues
- 📊 **Risk score** (0–100) for quick dataset health at a glance
- 🌊 **Drift detection** between reference and new datasets using PSI + total variation
- 🧪 **KNN label-noise estimation** — catch mislabeled samples before training
- 🔗 **Leakage detection** via correlation + mutual information
- 🖥️ Fully usable as a **CLI tool** or **Python library**

No YAML configs. No dashboards to spin up. Just point it at a CSV.

---

## ✨ Features

| Category | Check | Severity |
|---|---|---|
| **Data Quality** | Missing values (count + ratio per column) | Medium / High |
| **Data Quality** | Duplicate rows | Medium |
| **Data Quality** | Duplicate column names | Medium |
| **Data Quality** | Constant / zero-variance features | Medium |
| **Targets** | Target column missing values | High |
| **Targets** | Class imbalance (majority class ≥ 90%) | High |
| **Features** | Numeric outliers via IQR bounds | Low / Medium |
| **Features** | High-cardinality categorical features | Low |
| **Features** | ID-like columns (uuid, hash, etc.) | Low |
| **Features** | Highly correlated feature pairs (≥ 0.95) | Medium |
| **Leakage** | Correlation-based target leakage (≥ 0.95) | Critical |
| **Leakage** | Mutual-information target leakage (≥ 0.95) | Critical |
| **Noise** | KNN-based label noise estimate | High |
| **Drift** | Numeric drift via PSI + normalized mean shift | — |
| **Drift** | Categorical drift via total variation distance | — |
| **Drift** | Missing / new column detection across datasets | — |

---

## 📦 Installation

```bash
pip install ml-dataset-doctor
```

Or from source:

```bash
git clone https://github.com/AaravMehta-07/ml-dataset-doctor.git
cd ml-dataset-doctor
pip install -e .
```

**Requirements:** Python ≥ 3.8, pandas, numpy, scikit-learn

---

## 🖥️ CLI Usage

### Basic scan

```powershell
dataset-doctor data.csv --target y
```

Example output:
```
Rows: 10000
Columns: 18
Risk Score: 65/100
Duplicates: 42
Constant Features: 2
Missing Columns: 3
Potential Leakage Features: 1
Label Noise: 0.087
Issues:
- [critical] Potential leakage in 1 features
- [high] Target appears heavily imbalanced
- [medium] Missing values in 3 columns
- [medium] 42 duplicate rows found
Auto-fix/review plan available
```

### Save full JSON report

```powershell
dataset-doctor data.csv --target y --json --report report.json
```

### Apply safe automatic fixes

```powershell
dataset-doctor data.csv --target y --auto-fix --output data_cleaned.csv --fix-log fix_log.json
```

### Compare drift between two datasets

```powershell
dataset-doctor reference.csv --target y --compare new.csv --json
```

---

## 🐍 Python API

```python
import pandas as pd
from dataset_doctor import scan, apply_fixes, compare_drift

df = pd.read_csv("data.csv")

# Run full audit
report = scan(df, target="y")
print(report)                    # Human-readable summary
print(report.risk_score())       # 0–100 score
print(report.issues())           # Structured issue list
report_json = report.to_json()   # Full JSON report

# Apply safe auto-fixes
clean_df, fix_log = apply_fixes(df, report.fix_plan)
clean_df.to_csv("data_cleaned.csv", index=False)

# Detect drift
new_df = pd.read_csv("new_data.csv")
drift = compare_drift(df, new_df)
print(drift["numeric"])       # PSI + mean shift per numeric column
print(drift["categorical"])   # Total variation per categorical column
```

### Report object

```python
report.results        # dict — raw check results (all 15+ checks)
report.fix_plan       # dict — auto-fix/review plan
report.risk_score()   # int — 0 to 100
report.issues()       # list[dict] — severity, code, message per issue
report.to_dict()      # full dict representation
report.to_json()      # JSON string
```

---

## 🔬 What Gets Detected

### 🎯 Risk Score
Every issue is weighted by severity. The final score is capped at 100:

| Severity | Weight |
|---|---|
| `low` | +5 |
| `medium` | +15 |
| `high` | +25 |
| `critical` | +40 |

### 🧨 Target Leakage
Flags features that are suspiciously correlated with the target label:
- **Pearson correlation ≥ 0.95** → method: `"correlation"`
- **Mutual information ≥ 0.95** → method: `"mutual_info"` (classification + regression)

### 🏷️ Label Noise Estimation
Uses a **KNN classifier** (k=5) trained on numeric features to predict labels. Discrepancy between predictions and ground truth is reported as an estimated noise rate.

### 📈 Outlier Detection
IQR-based: values outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` are flagged with count, ratio, and bounds per column.

### 🌊 Drift Detection
| Metric | Applied To | What It Measures |
|---|---|---|
| **PSI** (Population Stability Index) | Numeric | Distribution shift across 10 quantile bins |
| **Normalized Mean Shift** | Numeric | Mean change normalized by reference std dev |
| **Total Variation Distance** | Categorical | Half the L1 distance between value distributions |
| **Missing/New Columns** | All | Columns present in one dataset but not the other |

---

## 🔧 Auto-Fix Engine

The fix plan is generated automatically from scan results and can be applied with `apply_fixes()`.

| Fix | Trigger |
|---|---|
| `drop_duplicates` | Duplicate rows detected |
| `drop_constant_features` | Zero-variance columns found |
| `impute_missing` (median) | Numeric columns with missing values |
| `impute_missing` (mode) | Categorical columns with missing values |
| `drop_missing_target` | Target column has NaN rows |
| `cap_outliers` (IQR clip) | Outliers detected in numeric columns |
| `review_*` flags | ID-like, high-cardinality, leakage, correlated features (human review) |

All fixes are **non-destructive** — they operate on a copy of the dataframe and return both the cleaned dataframe and a structured fix log.

---

## 🏗️ Architecture

```
ml-dataset-doctor/
├── dataset_doctor/
│   ├── __init__.py      # Public API: scan, apply_fixes, compare_drift
│   ├── checks.py        # All 15+ diagnostic checks + drift_check
│   ├── core.py          # scan(), apply_fixes(), compare_drift() orchestrators
│   ├── planner.py       # build_fix_plan() — generates actionable fix plan from results
│   ├── fixes.py         # apply_fix_plan() — executes fixes, returns (df, log)
│   ├── report.py        # Report class — risk_score, issues, to_json, __str__
│   └── cli.py           # CLI entry point (argparse → scan → output)
├── tests/
│   └── test_dataset_doctor.py
├── setup.py
└── requirements.txt
```

**Data flow:**
```
CSV / DataFrame
      │
      ▼
  checks.run_all_checks()       ← 15+ individual check functions
      │
      ▼
  planner.build_fix_plan()      ← generates safe fix actions
      │
      ▼
  Report(results, fix_plan)     ← risk_score + issues + to_json
      │
      ├──→ fixes.apply_fix_plan()   ← optional: clean DataFrame + log
      └──→ checks.drift_check()     ← optional: drift vs. new dataset
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/
```

All three test scenarios are covered:
- Categorical target scanning + fix plan generation
- Auto-fix imputation (numeric median + categorical mode)
- Drift detection with mismatched and categorical columns

---

## 📋 Full CLI Reference

```
usage: dataset-doctor [-h] --target TARGET [--json] [--report REPORT]
                      [--auto-fix] [--output OUTPUT] [--fix-log FIX_LOG]
                      [--compare COMPARE]
                      csv

positional arguments:
  csv                   CSV file to scan

options:
  --target TARGET       Target/label column (required)
  --json                Print full JSON report to stdout
  --report REPORT       Write JSON report to this file path
  --auto-fix            Apply safe automatic fixes
  --output OUTPUT       Output CSV for --auto-fix (default: data_cleaned.csv)
  --fix-log FIX_LOG     Fix log JSON path for --auto-fix (default: fix_log.json)
  --compare COMPARE     New CSV to compare for drift against the input CSV
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or PRs for:
- New check types (e.g., schema validation, text column detection)
- HTML/Markdown report output
- Additional drift metrics (KS test, JS divergence)
- Pandas 2.x / Polars support

```bash
git clone https://github.com/AaravMehta-07/ml-dataset-doctor.git
cd ml-dataset-doctor
pip install -e .
pytest tests/
```

---

## 📄 License

MIT © [Aarav Mehta](https://github.com/AaravMehta-07)

---

<p align="center">
  Built with ❤️ for ML engineers who care about data quality.<br/>
  <b>Star ⭐ the repo if it saved your model from bad data!</b>
</p>