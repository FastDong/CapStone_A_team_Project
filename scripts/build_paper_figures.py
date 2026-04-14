from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "ml" / "outputs" / "metrics"
REG_PLOTS_DIR = ROOT / "ml" / "outputs" / "plots" / "regression"
CLS_PLOTS_DIR = ROOT / "ml" / "outputs" / "plots" / "classification"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dirs() -> None:
    REG_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CLS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_regression_benchmark(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=False)
    palette = {
        "ridge": "#4c78a8",
        "xgboost_ref": "#f58518",
        "random_forest": "#54a24b",
        "extra_trees": "#e45756",
        "lightgbm": "#72b7b2",
    }

    for ax, target in zip(axes, ["HE_glu", "HE_sbp", "HE_chol"]):
        target_rows = [r for r in rows if r["target"] == target]
        target_rows.sort(key=lambda r: float(r["RMSE"]))
        models = [r["model"] for r in target_rows]
        rmses = [float(r["RMSE"]) for r in target_rows]
        colors = [palette.get(model, "#999999") for model in models]
        y = np.arange(len(models))
        ax.barh(y, rmses, color=colors)
        ax.set_yticks(y, models)
        ax.invert_yaxis()
        ax.set_title(target)
        ax.set_xlabel("RMSE")
        ax.grid(axis="x", alpha=0.25)
        for idx, row in enumerate(target_rows):
            label = f"R2={float(row['R2']):.3f}"
            ax.text(float(row["RMSE"]) + 0.03, idx, label, va="center", fontsize=8)

    fig.suptitle("Regression Model Benchmark", fontsize=13)
    save(fig, REG_PLOTS_DIR / "regression_model_benchmark_summary.png")


def build_regression_feature_importance(rows: list[dict[str, str]]) -> None:
    for target in ["HE_glu", "HE_sbp", "HE_chol"]:
        target_rows = [r for r in rows if r["target"] == target][:10]
        labels = [r["feature"] for r in target_rows][::-1]
        values = [float(r["importance"]) for r in target_rows][::-1]

        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        ax.barh(labels, values, color="#3d5a80")
        ax.set_title(f"{target} Feature Importance (Top 10)")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
        save(fig, REG_PLOTS_DIR / f"feature_importance_{target}.png")


def build_classification_default_summary(rows: list[dict[str, str]]) -> None:
    default_rows = [r for r in rows if r["threshold_type"] == "default_0.5"]
    models = [r["model_name"] for r in default_rows]
    accuracy = [float(r["accuracy"]) for r in default_rows]
    f1 = [float(r["f1"]) for r in default_rows]
    auc = [float(r["roc_auc"]) for r in default_rows]

    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x - width, accuracy, width, label="Accuracy", color="#98c1d9")
    ax.bar(x, f1, width, label="F1", color="#ee6c4d")
    ax.bar(x + width, auc, width, label="ROC-AUC", color="#3d5a80")
    ax.set_xticks(x, models)
    ax.set_ylim(0.45, 0.98)
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance at Default Threshold 0.5")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, CLS_PLOTS_DIR / "classification_default_summary.png")


def build_classification_threshold_summary(rows: list[dict[str, str]]) -> None:
    optimized_rows = [r for r in rows if r["threshold_type"] == "optimized"]
    models = [r["model_name"] for r in optimized_rows]
    thresholds = [float(r["threshold_value"]) for r in optimized_rows]
    f1 = [float(r["f1"]) for r in optimized_rows]

    fig, ax1 = plt.subplots(figsize=(8.8, 4.6))
    x = np.arange(len(models))
    bars = ax1.bar(x, f1, color="#3d5a80", width=0.55)
    ax1.set_xticks(x, models)
    ax1.set_ylim(0.55, 0.75)
    ax1.set_ylabel("Optimized F1")
    ax1.set_title("Optimized Threshold Results")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, thresholds, color="#e07a5f", marker="o", linewidth=2)
    ax2.set_ylim(0.45, 0.7)
    ax2.set_ylabel("Threshold")

    for bar, thr in zip(bars, thresholds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"thr={thr:.2f}", ha="center", va="bottom", fontsize=8)

    save(fig, CLS_PLOTS_DIR / "classification_optimized_summary.png")


def main() -> None:
    ensure_dirs()
    benchmark_rows = read_csv(METRICS_DIR / "regression_model_benchmark.csv")
    importance_rows = read_csv(METRICS_DIR / "feature_importance_regression.csv")
    classification_rows = read_csv(METRICS_DIR / "classification_metrics_best_regression_models.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    build_regression_benchmark(benchmark_rows)
    build_regression_feature_importance(importance_rows)
    build_classification_default_summary(classification_rows)
    build_classification_threshold_summary(classification_rows)

    print("Paper figures updated.")


if __name__ == "__main__":
    main()
