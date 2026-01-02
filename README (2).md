# 🩺 ml-dataset-doctor

Diagnose, fix, and audit machine learning datasets **before training**.

`ml-dataset-doctor` is a lightweight Python library designed to catch **data quality issues that silently break ML models**. It scans datasets, generates **safe auto-fix plans**, applies fixes with **full audit logs**, and supports both **CLI and Python API** workflows.

---

## 🚀 Why ml-dataset-doctor?

Most ML failures are caused by **bad data**, not bad models:
- hidden duplicates
- silent leakage
- extreme outliers
- noisy labels
- drifting distributions

`ml-dataset-doctor` helps you **detect and fix these issues early**, before wasting time training models.

---

## ✨ Features

### 🔍 Dataset Health Checks
- Missing value detection
- Duplicate row detection
- Constant / zero-variance feature detection
- Outlier detection (IQR-based)
- Label noise estimation
- Target leakage detection

### 🛠 Auto-Fix System (Safe by Design)
- Generates an **explicit fix plan**
- Applies fixes only when requested
- No silent data changes
- Full audit logs for every action

### 📉 Dataset Drift Detection
- Compare reference vs new datasets
- Detect numeric distribution shifts
- Useful for monitoring production data

### 🖥 Multiple Interfaces
- **Python API** for notebooks & pipelines
- **CLI tool** for quick checks and automation

---

## 📦 Installation

```bash
pip install ml-dataset-doctor
```

Requires:
- Python ≥ 3.8
- pandas
- numpy
- scikit-learn

---

## ⚡ Quick Start (Python)

```python
import pandas as pd
from dataset_doctor import scan, apply_fixes

df = pd.read_csv("data.csv")

# Scan dataset
report = scan(df, target="y")

print(report)
print(report.fix_plan)

# Apply fixes explicitly
clean_df, fix_log = apply_fixes(df, report.fix_plan)
```

---

## 🖥 CLI Usage

```bash
dataset-doctor data.csv --target y
dataset-doctor data.csv --target y --auto-fix
```

---

## 📉 Dataset Drift Comparison

```python
from dataset_doctor import compare_drift
import pandas as pd

train_df = pd.read_csv("train.csv")
new_df = pd.read_csv("new_data.csv")

drift = compare_drift(train_df, new_df)
print(drift)
```

---

## 🔒 Auto-Fix Philosophy

`ml-dataset-doctor` **never modifies data silently**.

✔ Fixes are planned first  
✔ You decide whether to apply them  
✔ Targets are never auto-modified  
✔ Every fix is logged  

---

## 📄 License

MIT License
