# 2026-03-28 연구 성과 브리핑

## 목적
- 비침습 입력(영양섭취 라벨, 신체계측, 연령)을 기반으로 대사증후군 분류 모델 성능 점검
- 상관관계 분석과 연령대별 성능 편차를 확인하여 후속 개선 방향 제시

## 핵심 결과
- 모델: XGBoost (GridSearchCV, ROC-AUC 기준 선택)
- 기본 임계값(0.5): Accuracy 0.8092, F1 0.5430, Recall 0.4813, ROC-AUC 0.8655
- F1 최적 임계값(0.29): F1 0.6519, Recall 0.8069, Precision 0.5469

## 시각화 산출물
- `feature_correlation_heatmap.png`: 피처-타깃 상관구조 확인
- `age_group_performance.png`: 연령대별 F1/Recall/ROC-AUC 비교
- `age_group_performance.csv`: 연령대별 지표 원본 표
- `xgboost_metrics.json`: 최종 모델 지표 및 최적 파라미터

## 관찰 포인트
- 고연령 구간으로 갈수록 Recall/F1은 상승하는 경향
- 반대로 ROC-AUC는 고연령에서 낮아지는 경향
- 단일 임계값보다 목적(선별/정밀)에 따라 임계값 운영 전략이 중요

