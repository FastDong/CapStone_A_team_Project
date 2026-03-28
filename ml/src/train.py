from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().parent / "pipeline" / "step4_train_xgboost.py"
    subprocess.run([sys.executable, str(script_path)], check=True)
    print("[DONE] training completed")


if __name__ == "__main__":
    main()
