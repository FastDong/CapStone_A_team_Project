# ML Workflow (2-Stage Regression -> Classification)

This implementation is split into two scripts:

1. `ml/src/train_regression.py`
- Loads `data/processed/final_dataset_with_label.csv`
- Applies preprocessing (sex encoding, missing summary, train-only imputation)
- Trains 3 regression models for health indicators: `HE_glu`, `HE_sbp`, `HE_chol`
- Compares target-specific settings:
  - per-target hyperparameter sets
  - optional `log1p` target transform
  - `reg:squarederror` vs `reg:pseudohubererror`
  - engineered features such as BMI, macro-energy ratios, and body-composition interactions
  - global vs sex-group-aware regression
- Generates out-of-fold train predictions (`pred_glu`, `pred_sbp`, `pred_chol`) to avoid leakage
- Saves baseline-vs-improved regression metrics, experiment tables, test prediction tables, scatter/residual plots, and the stage-2 input dataset

2. `ml/src/train_classification.py`
- Loads `ml/outputs/data/stage2_input_with_predictions.csv`
- Trains and compares:
  - Model 1: body-only (`WHtR`, `HE_wt`, `age`, `sex`)
  - Model 2: body + predicted health indicators (`pred_*` + body)
  - Model 3: body + actual health indicators (`HE_*` + body)
- Saves classification metrics, confusion matrices, ROC curve, feature importances, and final predictions

## Run

From repository root:

```bash
conda run -n machine python ml/src/train_regression.py
conda run -n machine python ml/src/train_classification.py
```

## Main Outputs

Outputs are grouped to keep the folder readable:

- `ml/outputs/metrics/`
  - `regression_metrics.csv`
  - `regression_baseline_vs_improved.csv`
  - `regression_experiment_results.csv`
  - `classification_metrics.csv`
  - `classification_metrics_threshold_optimized.csv`
  - `feature_importance_regression.csv`
  - `feature_importance_classification.csv`
  - `threshold_table_*.csv`
- `ml/outputs/data/`
  - `stage2_input_with_predictions.csv`
  - `regression_test_predictions.csv`
  - `final_predictions.csv`
- `ml/outputs/plots/regression/`
  - `actual_vs_predicted_HE_glu.png`
  - `actual_vs_predicted_HE_sbp.png`
  - `actual_vs_predicted_HE_chol.png`
  - `residual_plot_HE_glu.png`
  - `residual_plot_HE_sbp.png`
  - `residual_plot_HE_chol.png`
- `ml/outputs/plots/classification/`
  - `roc_curve.png`
  - `confusion_matrix_*.png`
  - `pr_curve_*.png`
  - `threshold_f1_*.png`
  - `feature_importance_*_classifier.png`
- `ml/outputs/metadata/`
  - `split_meta.json`
  - `feature_medians_train.csv`
  - `sex_mode_train.csv`
  - `regression_selected_configs.json`

This project is for preventive risk screening support and health-management guidance, not medical diagnosis.
