# ML Pipeline TODO (Sequence-Based)

이 문서는 `nutrient_2019.csv` + `NHANES_2017_2023.csv`를 직접 병합하기 어려운 상황을 전제로,
라벨 전이 기반으로 비침습 대사증후군 예측 모델을 만드는 작업 순서를 정의한다.

## 1. 데이터 준비
- [x] `data/raw/NHANES_2017_2023.csv` 배치
- [x] `data/raw/nutrient_2019.csv` 배치
- [ ] 컬럼 사전(의미/단위) 정리 문서화

## 2. 영양상태 생성 (nutrient_2019)
- [x] 개인 단위 집계(`ID` 기준) 생성
- [x] KNN + Softmax 기반 3클래스 라벨 생성
- [x] 결과 저장: `data/interim/nutrient_2019_person_level_with_status.csv`
- [ ] class 분포 불균형 점검 및 기준 튜닝

## 3. NHANES 영양섭취 라벨 전이
- [x] 공통 프로필 컬럼 교집합 기준 KNN 학습
- [x] NHANES 대상 `nutrition_intake`(0/1/2) 부여
- [x] 결과 저장: `data/interim/nhanes_with_nutrition_intake_label.csv`
- [ ] 연도별/성별 라벨 분포 안정성 점검

## 4. ATP III 라벨링
- [x] 기준 항목: 허리둘레, TG, HDL, 혈압, 공복혈당
- [x] `>=3` 항목 충족 시 대사증후군 라벨 1
- [x] 결과 저장: `data/processed/metabolic_syndrome_labeled_dataset.csv`
- [ ] cut-off를 한국인 기준(예: 허리둘레)으로 바꾼 민감도 분석

## 5. XGBoost 학습
- [x] 피처: `nutrition_intake`, `sex`, `HE_ht`, `HE_wt`, `HE_wc`, `age`
- [x] 타깃: `metabolic_syndrome`
- [x] 모델/지표 저장
- [ ] 하이퍼파라미터 탐색(CV) 추가

## 6. 재현성 / 보고서
- [x] 단계별 실행 파일 분리
- [x] README 실행 순서 반영
- [ ] 실험 로그 템플릿(`docs/experiments/`) 추가
- [ ] 논문용 방법론(라벨 전이 전략, 한계, 윤리/비진단 고지) 문서화

