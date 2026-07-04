import json


class Report:
    def __init__(self, results, fix_plan):
        self.results = results
        self.fix_plan = fix_plan

    def to_dict(self):
        return {"results": self.results, "fix_plan": self.fix_plan, "risk_score": self.risk_score(), "issues": self.issues()}

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, default=str)

    def issues(self):
        issues = []
        r = self.results
        if r.get("duplicates", 0):
            issues.append({"severity": "medium", "code": "duplicates", "message": f"{r['duplicates']} duplicate rows found"})
        if r.get("constant_features"):
            issues.append({"severity": "medium", "code": "constant_features", "message": f"{len(r['constant_features'])} constant features found"})
        if r.get("missing_summary"):
            high_missing = [c for c, v in r["missing_summary"].items() if v["ratio"] >= 0.3]
            severity = "high" if high_missing else "medium"
            issues.append({"severity": severity, "code": "missing_values", "message": f"Missing values in {len(r['missing_summary'])} columns"})
        if r.get("target_health", {}).get("missing", 0):
            issues.append({"severity": "high", "code": "target_missing", "message": "Target column contains missing labels"})
        if r.get("target_health", {}).get("majority_class_ratio", 0) >= 0.9 and r.get("target_health", {}).get("unique", 0) > 1:
            issues.append({"severity": "high", "code": "class_imbalance", "message": "Target appears heavily imbalanced"})
        if r.get("leakage"):
            issues.append({"severity": "critical", "code": "target_leakage", "message": f"Potential leakage in {len(r['leakage'])} features"})
        if r.get("feature_correlations"):
            issues.append({"severity": "medium", "code": "correlated_features", "message": f"{len(r['feature_correlations'])} highly correlated feature pairs"})
        if r.get("id_like_features"):
            issues.append({"severity": "low", "code": "id_like_features", "message": f"{len(r['id_like_features'])} ID-like columns may not generalize"})
        if r.get("high_cardinality_features"):
            issues.append({"severity": "low", "code": "high_cardinality", "message": f"{len(r['high_cardinality_features'])} high-cardinality categorical columns"})
        return issues

    def risk_score(self):
        weights = {"low": 5, "medium": 15, "high": 25, "critical": 40}
        return min(100, sum(weights.get(issue["severity"], 0) for issue in self.issues()))

    def summary(self):
        profile = self.results.get("profile", {})
        lines = [
            f"Rows: {profile.get('rows', self.results['shape'][0])}",
            f"Columns: {profile.get('columns', self.results['shape'][1])}",
            f"Risk Score: {self.risk_score()}/100",
            f"Duplicates: {self.results.get('duplicates', 0)}",
            f"Constant Features: {len(self.results.get('constant_features', []))}",
            f"Missing Columns: {len(self.results.get('missing_summary', {}))}",
            f"Potential Leakage Features: {len(self.results.get('leakage', {}))}",
            f"Label Noise: {self.results.get('label_noise')}",
        ]
        issues = self.issues()
        if issues:
            lines.append("Issues:")
            for issue in issues:
                lines.append(f"- [{issue['severity']}] {issue['message']}")
        if self.fix_plan:
            lines.append("Auto-fix/review plan available")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()