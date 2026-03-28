# Extended Abstract (English)

## Non-invasive Metabolic Syndrome Risk Classification with Nutrient Label Transfer and Threshold Optimization

This capstone study proposes a non-invasive machine learning pipeline for metabolic syndrome risk classification using easily obtainable inputs (age, sex, height, weight, waist circumference, and nutrition-intake status). A major challenge is that `NHANES_2017_2023.csv` and `nutrient_2019.csv` originate from different cohorts, making direct row-wise merging impractical.

To address this, we first build person-level nutrient status labels (deficient, balanced, excess) from `nutrient_2019` using a KNN+Softmax strategy. Since nutrient records include multiple entries per person per day, we aggregate at `ID + day` level and then use the per-person daily median to reduce outlier effects. We then transfer nutrient labels to NHANES subjects via profile-based KNN labeling.

Metabolic syndrome targets are defined using five criteria, with Korean waist cutoffs (male >= 90 cm, female >= 80 cm) and standard metabolic thresholds for triglycerides, HDL, blood pressure, and fasting glucose. Subjects meeting three or more criteria are labeled positive.

The final classifier is XGBoost with KNN-based missing value imputation and GridSearchCV (5-fold ROC-AUC) for hyperparameter selection. Input features are `nutrition_intake`, `nutrition_prob_2`, `waist_x_excess_prob`, `sex`, `HE_ht`, `HE_wt`, `HE_wc`, and `age`.

At the default threshold (0.50), test performance is Accuracy 0.8092, F1 0.5430, Recall 0.4813, and ROC-AUC 0.8655. Threshold optimization over 0.10-0.90 yields a best F1 threshold of 0.29, improving F1 to 0.6519 and Recall to 0.8069 (with lower accuracy 0.7970). This demonstrates that ranking quality is strong and decision-threshold tuning is critical in imbalanced screening tasks.

Age-stratified analysis shows increasing Recall/F1 with older age groups, while ROC-AUC tends to decrease. These findings suggest that group-specific calibration or threshold policies may further improve practical deployment.

This work is intended as a health management aid for research and educational use, not as a medical diagnostic system.

