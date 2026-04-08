# ML Workflow (2-Stage Regression -> Classification)

This implementation is split into two scripts:

1. `ml/src/train_regression.py`
- Loads `data/processed/final_dataset_with_label.csv`
- Applies preprocessing (sex encoding, missing summary, train-only imputation)
- Trains 3 regression models for health indicators: `HE_glu`, `HE_sbp`, `HE_chol`
- Generates out-of-fold train predictions (`pred_glu`, `pred_sbp`, `pred_chol`) to avoid leakage
- Saves regression metrics and stage-2 input dataset

2. `ml/src/train_classification.py`
- Loads `ml/outputs/stage2_input_with_predictions.csv`
- Trains and compares:
  - Model 1: body-only (`WHtR`, `HE_wt`, `age`, `sex`)
  - Model 2: body + predicted health indicators (`pred_*` + body)
  - Model 3: body + actual health indicators (`HE_*` + body)
- Saves classification metrics, confusion matrices, ROC curve, feature importances, and final predictions

## Run

From repository root:

```bash
python ml/src/train_regression.py
python ml/src/train_classification.py
```

## Main Outputs

- `ml/outputs/regression_metrics.csv`
- `ml/outputs/classification_metrics.csv`
- `ml/outputs/final_predictions.csv`
- `ml/outputs/feature_importance_regression.csv`
- `ml/outputs/feature_importance_classification.csv`
- `ml/outputs/roc_curve.png`
- `ml/outputs/confusion_matrix_body_only.png`
- `ml/outputs/confusion_matrix_predicted.png`
- `ml/outputs/confusion_matrix_actual.png`
- `ml/outputs/feature_importance_body_only_classifier.png`
- `ml/outputs/feature_importance_predicted_classifier.png`
- `ml/outputs/feature_importance_actual_classifier.png`

This project is for preventive risk screening support and health-management guidance, not medical diagnosis.
