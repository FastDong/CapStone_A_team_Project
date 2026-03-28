# ML Project

비침습 대사증후군 예측을 위한 전처리/라벨링/학습 파이프라인이다.

이 프로젝트는 진단 도구가 아니라 연구/캡스톤 목적의 건강관리 보조 가이드 모델이다.

## Input Data

- `../data/raw/NHANES_2017_2023.csv`
- `../data/raw/nutrient_2019.csv`

## Sequence Pipeline

1. `step1_build_nutrient_status.py`
- `nutrient_2019.csv`를 `ID + N_DAY(일자)` 단위로 1차 집계 후, 개인별 하루섭취 중앙값으로 집계
- 추가로 개인별 식사 패턴 스무딩 피처(`pattern_*`: 평균/변동성/과잉일 비율 등) 생성
- KNN + Softmax 기반으로 영양상태(0:부족, 1:균형, 2:과잉) 생성
- 출력: `../data/interim/nutrient_2019_person_level_with_status.csv`

2. `step2_label_nhanes_nutrition_intake.py`
- nutrient 데이터에서 학습한 패턴으로 NHANES에 `nutrition_intake` 라벨 전이
- KNN 이웃 가중평균으로 `pattern_*` 스무딩 피처도 함께 전이
- 출력: `../data/interim/nhanes_with_nutrition_intake_label.csv`

3. `step3_label_metabolic_syndrome_atpiii.py`
- ATP III 기준으로 대사증후군 라벨 생성
- 출력: `../data/processed/metabolic_syndrome_labeled_dataset.csv`

4. `step4_train_xgboost.py`
- 피처: `nutrition_intake`, `sex`, `HE_ht`, `HE_wt`, `HE_wc`, `age`
- 타깃: `metabolic_syndrome`
- 모델: XGBoost
- 출력:
  - `./models/xgboost_metabolic_syndrome.joblib`
  - `./outputs/xgboost_metrics.json`
  - `./outputs/xgboost_feature_importance.csv`

## Run

```bash
cd ml
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/predict.py
python src/analyze.py
```

## Directory Roles

- `src/pipeline/`: 단계별 전처리/라벨링/학습 스크립트
- `models/`: 저장된 학습 모델
- `outputs/`: 평가 결과 및 리포트
