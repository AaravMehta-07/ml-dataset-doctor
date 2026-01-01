import json

class Report:
    def __init__(self, results, fix_plan):
        self.results = results
        self.fix_plan = fix_plan

    def to_json(self):
        return json.dumps({
            "results": self.results,
            "fix_plan": self.fix_plan
        }, indent=2)

    def summary(self):
        lines = [
            f"Shape: {self.results['shape']}",
            f"Duplicates: {self.results['duplicates']}",
            f"Constant Features: {len(self.results['constant_features'])}",
            f"Label Noise: {self.results['label_noise']}",
        ]
        if self.results["leakage"]:
            lines.append("⚠️ Potential target leakage detected")
        if self.fix_plan:
            lines.append("Auto-fix plan available")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()
