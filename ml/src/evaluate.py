from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    metrics_file = root / "ml" / "outputs" / "xgboost_metrics.json"
    if not metrics_file.exists():
        print("No metrics file found. Run train.py first.")
        return
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
