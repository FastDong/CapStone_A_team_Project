from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


RANDOM_STATE = 42
OUTPUT_DIR = Path("ml/outputs")
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
CLASSIFICATION_PLOTS_DIR = PLOTS_DIR / "classification"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
INPUT_FILE = DATA_OUTPUT_DIR / "stage2_input_with_predictions.csv"
MODEL_DIR = Path("ml/models")
CLS_TARGET = "MetS_Label"

MODEL_SPECS = [
    {
        "model_name": "body_only",
        "features": ["WHtR", "HE_wt", "age", "sex"],
        "prob_col": "prob_body_only",
        "pred_default_col": "pred_body_only_default",
        "pred_opt_col": "pred_body_only_optimized",
        "threshold_plot": CLASSIFICATION_PLOTS_DIR / "threshold_f1_body_only.png",
        "pr_plot": CLASSIFICATION_PLOTS_DIR / "pr_curve_body_only.png",
        "confusion_plot": CLASSIFICATION_PLOTS_DIR / "confusion_matrix_body_only.png",
        "fi_plot": CLASSIFICATION_PLOTS_DIR / "feature_importance_body_only_classifier.png",
        "model_file": MODEL_DIR / "classifier_body_only.json",
    },
    {
        "model_name": "body_plus_predicted",
        "features": ["pred_glu", "pred_sbp", "pred_chol", "WHtR", "HE_wt", "age", "sex"],
        "prob_col": "prob_body_plus_predicted",
        "pred_default_col": "pred_body_plus_predicted_default",
        "pred_opt_col": "pred_body_plus_predicted_optimized",
        "threshold_plot": CLASSIFICATION_PLOTS_DIR / "threshold_f1_body_plus_predicted.png",
        "pr_plot": CLASSIFICATION_PLOTS_DIR / "pr_curve_body_plus_predicted.png",
        "confusion_plot": CLASSIFICATION_PLOTS_DIR / "confusion_matrix_predicted.png",
        "fi_plot": CLASSIFICATION_PLOTS_DIR / "feature_importance_predicted_classifier.png",
        "model_file": MODEL_DIR / "classifier_predicted_health_indicators.json",
    },
    {
        "model_name": "body_plus_actual",
        "features": ["HE_glu", "HE_sbp", "HE_chol", "WHtR", "HE_wt", "age", "sex"],
        "prob_col": "prob_body_plus_actual",
        "pred_default_col": "pred_body_plus_actual_default",
        "pred_opt_col": "pred_body_plus_actual_optimized",
        "threshold_plot": CLASSIFICATION_PLOTS_DIR / "threshold_f1_body_plus_actual.png",
        "pr_plot": CLASSIFICATION_PLOTS_DIR / "pr_curve_body_plus_actual.png",
        "confusion_plot": CLASSIFICATION_PLOTS_DIR / "confusion_matrix_actual.png",
        "fi_plot": CLASSIFICATION_PLOTS_DIR / "feature_importance_actual_classifier.png",
        "model_file": MODEL_DIR / "classifier_actual_health_indicators.json",
    },
]


def _print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _make_classifier(y_train: pd.Series) -> XGBClassifier:
    pos_cnt = int((y_train == 1).sum())
    neg_cnt = int((y_train == 0).sum())
    scale_pos_weight = float(neg_cnt / max(pos_cnt, 1))
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_oof_probabilities(x_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> np.ndarray:
    """
    Train-only OOF probabilities.
    Leakage prevention:
    - each sample's OOF probability is produced by a fold-model that did not see that sample.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = np.zeros(len(x_train), dtype=float)
    for tr_idx, val_idx in skf.split(x_train, y_train):
        x_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
        x_val = x_train.iloc[val_idx]
        model = _make_classifier(y_tr)
        model.fit(x_tr, y_tr)
        oof_prob[val_idx] = model.predict_proba(x_val)[:, 1]
    return oof_prob


def find_best_threshold(y_true: pd.Series, prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    thresholds = np.arange(0.10, 0.91, 0.01)
    rows: List[Dict[str, float]] = []
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        rows.append(
            {
                "threshold": float(round(thr, 2)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    table = pd.DataFrame(rows)
    # tie-break: higher f1 -> higher recall -> closer to 0.5
    best_row = max(
        rows,
        key=lambda r: (r["f1"], r["recall"], -abs(r["threshold"] - 0.5)),
    )
    return float(best_row["threshold"]), table


def evaluate_with_threshold(y_true: pd.Series, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold_value": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        # ROC-AUC is threshold-invariant
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


def plot_threshold_f1(table: pd.DataFrame, title: str, save_path: Path, best_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["threshold"], table["f1"], label="F1")
    ax.axvline(best_threshold, color="red", linestyle="--", label=f"best={best_threshold:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pr_curve(y_true: pd.Series, prob: np.ndarray, save_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion(y_true: pd.Series, pred: np.ndarray, title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(
    model: XGBClassifier,
    features: List[str],
    save_path: Path,
    title: str,
    model_name: str,
) -> pd.DataFrame:
    fi = pd.DataFrame({"model_name": model_name, "feature": features, "importance": model.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    top = fi.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fi


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    _print_section("1) Load Stage-2 Dataset")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run train_regression.py first.")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Data shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    _print_section("2) Missing Summary")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    _print_section("3) Recover Train/Test Split")
    train_df = df[df["set"] == "train"].copy()
    test_df = df[df["set"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Invalid split in stage2_input_with_predictions.csv.")
    train_df[CLS_TARGET] = pd.to_numeric(train_df[CLS_TARGET], errors="coerce").astype(int)
    test_df[CLS_TARGET] = pd.to_numeric(test_df[CLS_TARGET], errors="coerce").astype(int)
    y_train = train_df[CLS_TARGET]
    y_test = test_df[CLS_TARGET]
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print("Train MetS ratio:", y_train.value_counts(normalize=True).sort_index().to_dict())
    print("Test MetS ratio :", y_test.value_counts(normalize=True).sort_index().to_dict())

    metrics_rows: List[Dict[str, float]] = []
    fi_all: List[pd.DataFrame] = []
    prediction_cols = test_df[["row_id", "ID", CLS_TARGET]].copy()
    roc_inputs = []
    optimal_thresholds = {}
    f1_gains = {}

    _print_section("4) Train + Threshold Optimization per Model")
    for spec in MODEL_SPECS:
        name = spec["model_name"]
        features = spec["features"]

        print(f"\n[{name}] features: {features}")
        x_train_raw = train_df[features].copy()
        x_test_raw = test_df[features].copy()
        for c in features:
            x_train_raw[c] = pd.to_numeric(x_train_raw[c], errors="coerce")
            x_test_raw[c] = pd.to_numeric(x_test_raw[c], errors="coerce")

        # Fit imputer on train only, then transform both sets.
        imputer = SimpleImputer(strategy="median")
        x_train = pd.DataFrame(imputer.fit_transform(x_train_raw), columns=features, index=x_train_raw.index)
        x_test = pd.DataFrame(imputer.transform(x_test_raw), columns=features, index=x_test_raw.index)

        # Threshold selection must use train-internal predictions only.
        oof_prob = get_oof_probabilities(x_train, y_train, n_splits=5)
        best_thr, threshold_table = find_best_threshold(y_train, oof_prob)
        optimal_thresholds[name] = best_thr
        threshold_table.to_csv(METRICS_DIR / f"threshold_table_{name}.csv", index=False, encoding="utf-8-sig")
        plot_threshold_f1(
            table=threshold_table,
            title=f"Threshold vs F1 ({name})",
            save_path=spec["threshold_plot"],
            best_threshold=best_thr,
        )
        print(f"- {name} optimal threshold: {best_thr:.2f}")

        # Retrain with full train and evaluate on untouched test.
        model = _make_classifier(y_train)
        model.fit(x_train, y_train)
        test_prob = model.predict_proba(x_test)[:, 1]
        pred_default = (test_prob >= 0.5).astype(int)
        pred_opt = (test_prob >= best_thr).astype(int)

        m_default = evaluate_with_threshold(y_test, test_prob, 0.5)
        m_opt = evaluate_with_threshold(y_test, test_prob, best_thr)
        metrics_rows.append(
            {"model_name": name, "threshold_type": "default_0.5", **m_default}
        )
        metrics_rows.append(
            {"model_name": name, "threshold_type": "optimized", **m_opt}
        )
        f1_gains[name] = m_opt["f1"] - m_default["f1"]

        prediction_cols[spec["prob_col"]] = test_prob
        prediction_cols[spec["pred_default_col"]] = pred_default
        prediction_cols[spec["pred_opt_col"]] = pred_opt

        plot_pr_curve(
            y_true=y_test,
            prob=test_prob,
            save_path=spec["pr_plot"],
            title=f"Precision-Recall Curve ({name})",
        )
        plot_confusion(
            y_true=y_test,
            pred=pred_opt,
            title=f"Confusion Matrix ({name}, optimized threshold)",
            save_path=spec["confusion_plot"],
        )
        fi_all.append(
            plot_feature_importance(
                model=model,
                features=features,
                save_path=spec["fi_plot"],
                title=f"Feature Importance ({name})",
                model_name=name,
            )
        )

        model.save_model(str(spec["model_file"]))
        roc_inputs.append((name, test_prob, m_opt["roc_auc"]))

    _print_section("5) Save Metrics / Predictions / Curves")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        METRICS_DIR / "classification_metrics_threshold_optimized.csv",
        index=False,
        encoding="utf-8-sig",
    )
    # compatibility export
    metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False, encoding="utf-8-sig")

    prediction_cols.to_csv(DATA_OUTPUT_DIR / "final_predictions.csv", index=False, encoding="utf-8-sig")

    fi_df = pd.concat(fi_all, ignore_index=True)
    fi_df.to_csv(METRICS_DIR / "feature_importance_classification.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name, prob, auc_val in roc_inputs:
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (3 Models)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CLASSIFICATION_PLOTS_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    print(f"Saved: {METRICS_DIR / 'classification_metrics_threshold_optimized.csv'}")
    print(f"Saved: {METRICS_DIR / 'classification_metrics.csv'}")
    print(f"Saved: {DATA_OUTPUT_DIR / 'final_predictions.csv'}")
    print(f"Saved: {METRICS_DIR / 'feature_importance_classification.csv'}")
    print(f"Saved: {CLASSIFICATION_PLOTS_DIR / 'roc_curve.png'}")
    for spec in MODEL_SPECS:
        print(f"Saved: {spec['threshold_plot']}")
        print(f"Saved: {spec['pr_plot']}")
        print(f"Saved: {spec['confusion_plot']}")
        print(f"Saved: {spec['fi_plot']}")

    _print_section("6) Paper-Oriented Summary")
    for name in ["body_only", "body_plus_predicted", "body_plus_actual"]:
        print(f"- {name} optimal threshold: {optimal_thresholds[name]:.2f}")
    print("\nF1 gain (optimized - default_0.5):")
    for name in ["body_only", "body_plus_predicted", "body_plus_actual"]:
        print(f"- {name}: {f1_gains[name]:+.4f}")

    def _get_metric(model_name: str, thr_type: str, metric: str) -> float:
        row = metrics_df[(metrics_df["model_name"] == model_name) & (metrics_df["threshold_type"] == thr_type)].iloc[0]
        return float(row[metric])

    print("\nPerformance progression (default -> optimized):")
    for name in ["body_only", "body_plus_predicted", "body_plus_actual"]:
        f1_d = _get_metric(name, "default_0.5", "f1")
        f1_o = _get_metric(name, "optimized", "f1")
        print(f"- {name}: F1 {f1_d:.4f} -> {f1_o:.4f}")

    print("\nROC-AUC note:")
    print("- ROC-AUC is computed from probabilities and does not depend on threshold.")
    print("- Threshold optimization mainly shifts precision/recall trade-off, which changes F1.")


if __name__ == "__main__":
    main()
