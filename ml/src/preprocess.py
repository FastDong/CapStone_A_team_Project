from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / "pipeline" / script_name
    print(f"[RUN] {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    run_step("step1_build_nutrient_status.py")
    run_step("step2_label_nhanes_nutrition_intake.py")
    run_step("step3_label_metabolic_syndrome_atpiii.py")
    print("[DONE] preprocessing pipeline completed")


if __name__ == "__main__":
    main()
