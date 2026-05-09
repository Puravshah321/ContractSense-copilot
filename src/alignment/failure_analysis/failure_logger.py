"""
failure_logger.py
Captures failed predictions.
Logs prompt, model output, expected pattern, and stores them
in structured format for analysis.
"""

import json
import re
from pathlib import Path
from datetime import datetime


RISK_PATTERN = re.compile(r"^RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", re.MULTILINE)
ACTION_PATTERN = re.compile(r"ACTION:", re.MULTILINE)
CITATION_PATTERN = re.compile(r"CITATION:\s*\[.+?\]")


class FailureLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.failures = []
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def check_and_log(self, prompt: str, output: str, model_name: str = "unknown") -> bool:
        issues = []

        if not RISK_PATTERN.search(output):
            issues.append("missing_risk_label")

        if not ACTION_PATTERN.search(output):
            issues.append("no_action")

        if not CITATION_PATTERN.search(output):
            issues.append("missing_citation")

        if len(output.split()) > 500:
            issues.append("verbose_output")
        elif len(output.split()) < 10:
            issues.append("too_brief")

        if not issues:
            return False

        self.failures.append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt": prompt[:500],
            "output": output[:1000],
            "issues": issues,
            "output_length_tokens": len(output.split()),
        })

        return True

    def log_failure(self, prompt: str, output: str, expected_pattern: str,
                    model_name: str = "unknown", error_type: str = "unknown"):
        self.failures.append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt": prompt[:500],
            "output": output[:1000],
            "expected_pattern": expected_pattern,
            "error_type": error_type,
        })

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.failures, f, indent=2)
        print(f"💾 Saved {len(self.failures)} failure records to {self.log_path}")

    def get_failure_count(self) -> int:
        return len(self.failures)

    def get_summary(self) -> dict:
        from collections import Counter
        all_issues = []
        for f in self.failures:
            all_issues.extend(f.get("issues", [f.get("error_type", "unknown")]))
        return dict(Counter(all_issues))


def log_failures_from_dataset(
    dataset_path: str,
    output_path: str,
    model_name: str = "dpo_aligned",
):
    with open(dataset_path) as f:
        data = json.load(f)

    logger = FailureLogger(output_path)

    for entry in data:
        prompt = entry.get("prompt", "")
        chosen = entry.get("chosen", "")
        rejected = entry.get("rejected", "")

        logger.check_and_log(prompt, rejected, model_name=f"{model_name}_rejected")

    logger.save()

    summary = logger.get_summary()
    print(f"\n📊 Failure Summary for {model_name}:")
    for issue, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"   {issue}: {count}")

    return logger
