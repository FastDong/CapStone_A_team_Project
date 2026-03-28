# 2026-03-28 연구성과 브리핑 (최종)

## 1) 결론 요약
- 최종 운영 모델은 **Full 모델 + F1 최적 임계값(0.36)** 로 확정
- 이유: A/B 비교에서 Full 모델이 F1, AUC 모두 소폭 우세
- 선별 우선 시나리오는 Recall 우선 임계값(0.23)을 보조 정책으로 사용

## 2) 핵심 지표
- 기본 threshold 0.50  
  Accuracy `0.8109`, Precision `0.6319`, Recall `0.4720`, F1 `0.5404`, ROC-AUC `0.8664`
- F1 최적 threshold 0.36  
  Accuracy `0.8105`, Precision `0.5740`, Recall `0.7571`, F1 `0.6530`
- Recall 우선 threshold 0.23  
  Precision `0.5052`, Recall `0.8603`, F1 `0.6366`

## 3) A/B 비교 (Full vs Light)
| Model | Features | Default F1 | ROC-AUC | Best-F1 Threshold | Best-F1 |
|---|---:|---:|---:|---:|---:|
| Full | 18 | 0.5404 | 0.8664 | 0.36 | 0.6530 |
| Light | 11 | 0.5364 | 0.8662 | 0.34 | 0.6517 |

### 해석
- Full 모델이 성능상 근소 우세
- Light 모델은 피처 단순화(11개) 대비 성능 손실이 작아 배포 경량 대안으로 가능

## 4) 생활패턴 스무딩 기법
영양 데이터(`nutrient_2019`)를 “한 달 패턴 유사” 구조로 반영하기 위해 다음 과정을 적용함.
1. `ID + N_DAY(+year)` 단위로 하루 섭취량 합산
2. 개인 단위에서 일일 분포 통계(평균, 표준편차, IQR) 계산
3. 과잉/부족/균형 일수 비율 계산
4. 단위 열량당 영양밀도(단백질/지방/탄수/나트륨) 계산
5. KNN 이웃 가중평균으로 NHANES 대상자에게 스무딩 패턴 피처 전이

최종 Full 모델에 포함된 스무딩 피처:
- `pattern_energy_ratio_mean`
- `pattern_energy_ratio_std`
- `pattern_energy_ratio_iqr`
- `pattern_excess_day_ratio`
- `pattern_deficient_day_ratio`
- `pattern_balanced_day_ratio`
- `pattern_protein_density_mean`
- `pattern_fat_density_mean`
- `pattern_carb_density_mean`
- `pattern_sodium_density_mean`

## 5) 시각화/결과 파일
- `feature_correlation_heatmap.png`
- `age_group_performance.png`
- `age_group_performance.csv`
- `xgboost_metrics.json`
- `xgboost_ab_comparison.csv`
- `xgboost_threshold_metrics.csv`
- `xgboost_threshold_metrics_light.csv`

