from __future__ import annotations

import json
from pathlib import Path


def build_report(output_path: str = "benchmarks/local_eval_report.json") -> dict:
    report = {
        "metrics": {
            "mmAP": None,
            "mAP50": None,
            "mAP75": None,
        },
        "paper_baseline": {
            "URPC2021_A1": {"mmAP": 0.538, "mAP50": 0.919, "mAP75": 0.569},
            "URPC2022_A2": {"mmAP": 0.492, "mAP50": 0.954, "mAP75": 0.447},
        },
        "notes": "Metric hooks prepared. Connect real URPC evaluator on server data path.",
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
