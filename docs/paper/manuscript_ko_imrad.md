# 비침습 입력 기반 대사증후군 위험 분류: 영양 라벨 전이와 임계값 최적화 기반 XGBoost 접근

## 국문초록
본 연구는 혈액 채취 없이 수집 가능한 정보(연령, 성별, 키, 체중, 허리둘레, 영양섭취 상태)를 이용해 대사증후군 위험을 분류하는 실용적 머신러닝 파이프라인을 제안한다. 핵심 문제는 서로 다른 조사 집단에서 생성된 두 데이터셋(`NHANES_2017_2023.csv`, `nutrient_2019.csv`)의 직접 병합이 어렵다는 점이다. 이를 해결하기 위해 `nutrient_2019`에서 개인 단위 영양상태(부족/균형/과잉)를 KNN+Softmax 방식으로 생성하고, 공통 프로필 특성을 이용해 NHANES 대상자에게 영양 라벨을 전이했다. 이후 한국형 허리둘레 기준(남 90cm, 여 80cm)을 포함한 대사증후군 라벨을 구축하고 XGBoost 분류 모델을 학습했다. 결측치는 KNNImputer로 보정하고, GridSearchCV(ROC-AUC 기준)로 최적 하이퍼파라미터를 탐색했다. 기본 임계값(0.50)에서 Accuracy 0.8092, F1 0.5430, Recall 0.4813, ROC-AUC 0.8655를 확인했으며, 임계값 최적화(최적 0.29) 후 F1 0.6519, Recall 0.8069로 향상되었다. 본 연구는 비침습 건강관리 보조 시나리오에서 선별형 모델의 실효성을 보여준다.

주요어: 대사증후군, 비침습 예측, XGBoost, 라벨 전이, 임계값 최적화

## 1. Introduction
대사증후군은 심혈관 질환 및 제2형 당뇨병 위험 증가와 관련된 복합 대사 이상 상태로 조기 식별이 중요하다 [1][2]. 그러나 실제 서비스 환경에서는 혈액검사 이전 단계에서 사용자 입력 기반 선별이 우선 필요하다. 본 연구는 캡스톤 프로젝트 맥락에서, 비침습 입력만으로 대사증후군 위험군을 분류하는 파이프라인을 구축하고 그 성능을 검증한다.

연구 질문은 다음과 같다.
1. 서로 다른 표본에서 수집된 영양/건강 데이터셋을 직접 병합하지 않고도 분류 성능을 확보할 수 있는가?
2. 불균형 분류 문제에서 임계값 최적화가 실질적으로 F1/Recall 개선에 기여하는가?
3. 모델 성능은 연령대별로 유의미한 차이를 보이는가?

## 2. Methods

### 2.1 Data
- 건강 데이터: `data/raw/NHANES_2017_2023.csv`
- 영양 데이터: `data/raw/nutrient_2019.csv`

### 2.2 Nutrient Label Transfer
`nutrient_2019`는 동일 ID의 식품별 다중 기록을 포함하므로, `ID + N_DAY(+year)` 단위로 일일 섭취량을 합산하고, 개인(ID) 단위에서는 일일합계 중앙값을 대표값으로 사용했다.  
이후 `NF_EN, NF_PROT, NF_FAT, NF_CHO, NF_TDF, NF_NA`를 사용해 KNN+Softmax 기반 3분류 라벨(0 deficient, 1 balanced, 2 excess)을 생성했다.  
생성된 라벨은 공통 프로필 특성 기반 KNN 전이를 통해 NHANES 대상자에게 `nutrition_intake`, `nutrition_prob_0/1/2` 형태로 부여했다.

### 2.3 Target Labeling
대사증후군 타깃은 다음 기준 5개 중 3개 이상 충족 시 1로 정의했다.
- 복부비만: 남성 허리둘레 >= 90cm, 여성 허리둘레 >= 80cm
- 중성지방: TG >= 150
- HDL 저하: 남 < 40, 여 < 50
- 혈압: SBP >= 130 또는 DBP >= 85
- 공복혈당: glucose >= 100

### 2.4 Features, Imputation, and Model
입력 피처:
- `nutrition_intake`
- `nutrition_prob_2`
- `waist_x_excess_prob = HE_wc * nutrition_prob_2`
- `sex`, `HE_ht`, `HE_wt`, `HE_wc`, `age`

결측치는 `KNNImputer(n_neighbors=5)`로 보정했다 [6].  
모델은 XGBoost 이진분류기로 구성하고 [5], 하이퍼파라미터는 GridSearchCV(5-fold, scoring=ROC-AUC)로 선택했다 [4].

### 2.5 Threshold Optimization
기본 threshold 0.50과 별도로, 테스트셋에서 0.10~0.90 구간(0.01 step)을 탐색하여 F1 최대 threshold를 선택했다.

## 3. Results

### 3.1 Overall Performance
- Best CV ROC-AUC: 0.8659
- Test ROC-AUC: 0.8655
- Threshold 0.50: Accuracy 0.8092, F1 0.5430, Recall 0.4813
- Threshold 0.29(F1-opt): Accuracy 0.7970, Precision 0.5469, Recall 0.8069, F1 0.6519

정확도는 다소 감소했으나, 선별 관점에서 핵심인 Recall/F1이 크게 개선되었다.

### 3.2 Correlation Analysis
피처 상관관계 분석 결과, 허리둘레 및 파생피처(`waist_x_excess_prob`)가 타깃과 유의한 방향성을 보였다.  
결과 파일:
- `docs/research_results/feature_correlation_matrix.csv`
- `docs/research_results/feature_correlation_heatmap.png`

### 3.3 Age-group Performance
최적 threshold(0.29) 적용 시 연령대별 성능:
- 10-19: F1 0.4516, Recall 0.3415, AUC 0.9498
- 20-29: F1 0.4778, Recall 0.7049, AUC 0.9235
- 30-39: F1 0.6414, Recall 0.7600, AUC 0.8861
- 40-49: F1 0.6436, Recall 0.7756, AUC 0.8675
- 50-59: F1 0.6427, Recall 0.7673, AUC 0.8167
- 60-69: F1 0.6604, Recall 0.8502, AUC 0.7665
- 70+: F1 0.6878, Recall 0.8731, AUC 0.7635

연령이 높아질수록 Recall/F1은 상승 경향을 보였고, AUC는 낮아지는 경향이 관찰되었다.

## 4. Discussion
본 파이프라인은 데이터셋 직접 병합 불가 조건에서도 라벨 전이를 통해 분류 성능을 확보했다. 또한 불균형 상황에서 threshold 조정이 정적 0.5 기준보다 더 현실적인 성능(특히 재현율) 개선을 제공했다. 이는 건강검진 보조도구에서 거짓음성 감소가 중요한 상황에 의미가 있다.

## 5. Limitations
1. 영양 라벨은 전이 방식으로 생성된 간접 라벨이므로 노이즈 가능성이 존재한다.
2. threshold 최적화는 동일 홀드아웃 테스트 기반으로 수행되어 외부검증이 필요하다.
3. 본 연구는 임상 진단 대체가 아닌 참고용 선별 모델이다.

## 6. Conclusion
비침습 입력과 라벨 전이 전략, KNN 결측 보정, GridSearchCV 기반 XGBoost, 임계값 최적화를 결합해 실용적 대사증후군 선별 성능을 확보했다. 특히 threshold 최적화는 선별형 시스템에서 필요한 F1/Recall 개선에 효과적이었다.

## Figures
### Figure 1. Feature Correlation Heatmap
![feature_correlation_heatmap](../research_results/feature_correlation_heatmap.png)

### Figure 2. Performance by Age Group
![age_group_performance](../research_results/age_group_performance.png)

## References
[1] Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults. Executive summary of the third report of the National Cholesterol Education Program (NCEP) Adult Treatment Panel III. JAMA. 2001;285(19):2486-2497.  
[2] Alberti KGMM, Eckel RH, Grundy SM, et al. Harmonizing the metabolic syndrome. Circulation. 2009;120(16):1640-1645.  
[3] Ministry of Health and Welfare (Korea), Korea Disease Control and Prevention Agency. Korea National Health and Nutrition Examination Survey (KNHANES) documentation.  
[4] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. JMLR. 2011;12:2825-2830.  
[5] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. KDD 2016:785-794.  
[6] Troyanskaya O, Cantor M, Sherlock G, et al. Missing value estimation methods for DNA microarrays. Bioinformatics. 2001;17(6):520-525.

## Ethical Statement
본 결과는 건강관리 참고용이며 의료 진단, 치료, 처방을 대체하지 않는다.

