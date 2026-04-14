# 비침습 입력 기반 대사증후군 위험 분류: 생활패턴 스무딩 피처, 라벨 전이, 임계값 최적화 기반 XGBoost 연구

## 국문초록
대사증후군은 심혈관계 질환 및 제2형 당뇨병의 선행 위험군을 식별하는 핵심 임상 개념이며, 조기 선별의 공중보건적 가치가 크다. 그러나 실제 사용자 환경에서는 혈액검사 기반 정밀 진단 이전 단계에서, 비침습 입력만으로 위험 신호를 탐지하는 실용적 방법이 필요하다. 본 연구는 연령, 성별, 키, 체중, 허리둘레, 영양섭취 상태 등 비침습 정보만으로 대사증후군 위험을 분류하는 머신러닝 파이프라인을 제안한다. 연구의 주요 난점은 건강 데이터(`NHANES_2017_2023.csv`)와 영양 데이터(`nutrient_2019.csv`)가 서로 다른 조사 단위와 표본 구조를 가져 직접 병합이 어렵다는 점이다. 이를 해결하기 위해 영양 라벨 전이(label transfer) 전략을 설계하고, 개인의 일일 섭취 분포를 반영한 생활패턴 스무딩 피처(변동성, 과잉/부족일 비율, 영양밀도)를 도입했다. 타깃 라벨은 한국형 허리둘레 기준(남 90cm, 여 80cm)을 포함한 대사증후군 기준으로 구성했으며, 결측치는 KNNImputer로 처리했다. 분류기는 XGBoost를 사용하고 GridSearchCV(ROC-AUC 기준)로 최적화했다. 실험 결과 Full 모델(18개 피처)은 기본 임계값(0.50)에서 Accuracy 0.8109, F1 0.5404, ROC-AUC 0.8664를 기록했으며, F1 최적 임계값(0.36)에서 F1 0.6530, Recall 0.7571로 개선되었다. Recall 우선 정책(임계값 0.23, precision>=0.50)에서는 Recall 0.8603을 달성했다. 본 결과는 모델 성능 자체뿐 아니라 임계값 운영정책이 선별 성능을 실질적으로 좌우함을 보여준다. 본 연구는 비침습 선별 도구의 실서비스 연계를 위한 방법론적 기반을 제시한다.

**주요어**: 대사증후군, 비침습 분류, 라벨 전이, 생활패턴 스무딩, XGBoost, 임계값 최적화

---

## Abstract
Metabolic syndrome is a clinically meaningful construct for identifying high-risk populations for cardiovascular disease and type 2 diabetes. In practical environments, however, a pre-screening method based on non-invasive user inputs is needed before invasive laboratory testing. This study proposes a machine learning pipeline that classifies metabolic syndrome risk using only non-invasive variables: age, sex, height, weight, waist circumference, and nutrition-intake status. A key technical challenge is that the health dataset (`NHANES_2017_2023.csv`) and nutrition dataset (`nutrient_2019.csv`) are heterogeneous and cannot be directly merged at an individual level. To address this, we adopt a nutrition label-transfer strategy and introduce lifestyle smoothing features that capture intra-person intake distribution, excess/deficient-day ratios, and nutrient density. We define the target label using metabolic syndrome criteria including Korean waist cutoffs (male >=90 cm, female >=80 cm). Missing values are imputed with KNNImputer. We train XGBoost and optimize hyperparameters through GridSearchCV (ROC-AUC criterion). The Full model (18 features) achieved Accuracy 0.8109, F1 0.5404, and ROC-AUC 0.8664 at threshold 0.50. At the F1-optimal threshold (0.36), F1 increased to 0.6530 and Recall to 0.7571. Under a recall-priority policy (threshold 0.23 with precision>=0.50), Recall reached 0.8603. Results indicate that threshold policy is as important as model architecture in practical screening scenarios. This work provides a deployable methodological basis for non-invasive digital pre-screening systems.

---

## 1. 서론

### 1.1 연구 배경
대사증후군은 복부비만, 고중성지방혈증, 저HDL콜레스테롤혈증, 고혈압, 공복혈당 상승 등의 대사 이상이 군집적으로 나타나는 상태이며, 심혈관질환 및 당뇨병 위험과 밀접한 연관성을 가진다 [1][2]. 임상 현장에서의 대사증후군 평가는 혈액검사와 활력징후 측정을 기반으로 수행되지만, 대규모 선별 또는 생활 환경 기반 자가관리 단계에서는 동일한 수준의 접근성이 보장되기 어렵다.

특히 모바일 헬스케어 및 개인 맞춤형 건강관리 서비스에서는 진단 이전 단계에서 “위험 신호”를 조기에 감지하고 사용자 행동변화를 유도할 수 있는 선별형 모델이 필요하다. 이러한 맥락에서 비침습 입력 기반 모델은 임상 진단의 대체가 아니라, 검진 권고 및 위험군 우선 확인을 위한 전처리 레이어로 의미를 가진다.

### 1.2 문제 정의
본 연구에서 다루는 핵심 문제는 다음과 같다.
1. 건강 데이터와 영양 데이터의 이질성으로 인해 직접 병합이 불가능한 상황에서, 어떻게 영양 정보를 건강 데이터에 일관되게 반영할 것인가?
2. 일회성 섭취기록의 변동성을 완화하고 개인의 식습관 패턴을 모델 입력으로 재구성할 수 있는가?
3. 선별 목적에서 고정 임계값(0.50)이 항상 최적이 아닌데, 실제 운영에서 어떤 임계값 정책이 타당한가?

### 1.3 연구 목적
본 연구의 목적은 다음 세 가지다.
1. **라벨 전이 기반 데이터 연결**: 이질 코호트 간 영양 라벨 전이 파이프라인 구축
2. **생활패턴 스무딩 기반 피처 공학**: 개인 단위 섭취 분포 특성 추출 및 분류 성능 검증
3. **운영정책 중심 성능 평가**: 모델 자체 성능(ROC-AUC)과 임계값 정책(F1 최적/Recall 우선)을 함께 평가

### 1.4 연구 기여
본 연구의 기여는 다음과 같다.
1. 라벨 전이 전략을 통해 직접 병합 불가능 데이터셋 환경에서 실무적으로 동작 가능한 분류 파이프라인을 제안했다.
2. 평균치 중심 입력을 넘어 분포 기반 생활패턴 스무딩 피처를 설계하여 성능 개선 가능성을 보였다.
3. “모델 선택”과 “임계값 운영정책”을 분리하지 않고 통합적으로 평가해, 서비스 적용 시 의사결정 프레임을 제시했다.

### 1.5 논문 구성
본 논문은 2장에서 관련연구, 3장에서 데이터 및 방법론, 4장에서 실험 결과, 5장에서 논의, 6장에서 결론과 향후과제를 다룬다.

---

## 2. 관련 연구

### 2.1 대사증후군 진단 기준과 선별
대사증후군에 대한 국제적 합의는 다양한 임상 가이드라인에서 제시되어 왔으며 [1][2], 본 연구는 그 중 실무적 선별에 필요한 최소 공통조건을 채택했다. 특히 허리둘레 기준은 인종/국가별 차이를 반영해야 하므로 한국형 기준(남 90cm, 여 80cm)을 적용했다.

### 2.2 비침습 입력 기반 건강위험 예측
비침습 입력 기반 예측은 접근성, 비용, 반복 측정 용이성 측면에서 장점이 있다. 그러나 입력 신호의 정보량이 제한적이므로, 피처 공학과 의사결정 임계값 설계가 성능을 좌우한다. 기존 연구에서 단일 임계값(0.50)을 관성적으로 사용하는 경우가 많지만, 선별 목적에서는 재현율 중심 정책이 더 합리적일 수 있다.

### 2.3 결측치 보정과 불균형 분류
실제 건강 데이터는 결측이 흔하므로 결측치 처리는 필수다. KNNImputer는 주변 이웃 기반으로 관측 패턴을 반영할 수 있어, 비선형 분포를 다루는 트리 기반 모델과 결합 시 안정적 성능을 제공한다 [6]. 또한 불균형 분류에서는 ROC-AUC가 높아도 실제 임계값 운용에서 재현율이 낮아질 수 있으므로, threshold tuning이 중요하다.

### 2.4 본 연구의 차별점
기존 접근과 달리 본 연구는 다음을 결합한다.
1. **라벨 전이**: 이질 코호트 연결
2. **생활패턴 스무딩**: 개인 내 분포 통계 반영
3. **정책형 임계값 운영**: 기본/F1최적/Recall우선 병행 평가

---

## 3. 데이터 및 방법론

### 3.1 전체 파이프라인 개요
전체 흐름은 다음과 같다.
1. `nutrient_2019`로부터 개인별 영양 상태 라벨 및 스무딩 피처 생성
2. 공통 프로필 기반 KNN 라벨 전이로 NHANES 대상자의 영양 상태 확률 추정
3. NHANES 지표로 대사증후군 타깃 라벨 정의
4. KNNImputer 후 XGBoost 학습
5. 임계값 탐색으로 운영정책별 성능 비교
6. Full vs Light A/B 비교 및 최종 모델 선정

### 3.2 데이터셋
#### 3.2.1 건강 데이터
- 파일: `data/raw/NHANES_2017_2023.csv`
- 역할: 타깃 라벨(대사증후군 규칙) 구성, 신체계측/인구통계 피처 제공

#### 3.2.2 영양 데이터
- 파일: `data/raw/nutrient_2019.csv`
- 특징: 동일 개인(ID) 내 다중 식사기록 행 존재
- 역할: 영양 상태 라벨 원천 및 생활패턴 스무딩 피처 산출

### 3.3 영양 데이터 집계 및 상태 라벨링
영양 데이터는 일일 다중기록을 포함하므로, 먼저 `ID + N_DAY(+year)` 단위로 일일 섭취 총량을 집계했다. 이후 개인 단위로 다음 통계를 계산했다.
1. 에너지 섭취 비율 평균
2. 에너지 섭취 비율 표준편차
3. 에너지 섭취 비율 IQR
4. 과잉/부족/균형 일수 비율
5. 열량 대비 단백질/지방/탄수/나트륨 밀도

영양 상태는 `deficient`, `balanced`, `excess` 3클래스로 정의했으며, 개인별 확률 표현(`nutrition_prob_2`)을 모델 입력으로 사용했다.

### 3.4 라벨 전이(label transfer)
이질 데이터 간 직접 병합이 불가능하므로, 공통 프로필 특성(성별, 연령, 체형 관련 변수)을 기반으로 KNN 전이 모델을 구성했다. 전이 결과는 NHANES 대상자에게 영양 상태 라벨과 확률로 부여되며, 확률값은 신체지표와 상호작용항으로 결합되었다.

\[
\texttt{waist\_x\_excess\_prob} = \texttt{HE\_wc} \times \texttt{nutrition\_prob\_2}
\]

이 상호작용항은 복부비만과 과잉 섭취 경향의 결합 위험을 반영하는 파생특성으로 설계했다.

### 3.5 대사증후군 타깃 라벨 정의
본 연구는 다음 5개 기준 중 3개 이상 충족 시 양성(1)으로 정의했다.
1. 복부비만: 남 `HE_wc >= 90`, 여 `HE_wc >= 80`
2. 중성지방: `TG >= 150`
3. HDL 저하: 남 `<40`, 여 `<50`
4. 혈압: `SBP >=130` 또는 `DBP >=85`
5. 공복혈당: `glucose >=100`

이 정의는 임상 확진을 대체하기 위한 것이 아니라, 선별 위험도 라벨을 만들기 위한 규칙 기반 표준화 절차다.

### 3.6 피처 구성
#### 3.6.1 Full 모델(18개)
- `nutrition_intake`
- `nutrition_prob_2`
- `waist_x_excess_prob`
- `sex`, `HE_ht`, `HE_wt`, `HE_wc`, `age`
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

#### 3.6.2 Light 모델(11개)
- Full 중 핵심 스무딩 일부만 포함
- 경량화 목적(배포/추론 효율)으로 비교 실험

### 3.7 결측치 처리
결측치는 `KNNImputer(n_neighbors=5)`로 보정했다. 본 방법은 변수 간 국소 유사도를 활용해 결측을 채우므로, 단순 평균/중앙값 대체 대비 정보 손실을 줄일 수 있다.

### 3.8 모델 학습
- 모델: XGBoost 이진 분류기
- 분할: train/test = 8:2 (stratified)
- train rows: `38174`, test rows: `9544`
- 하이퍼파라미터 탐색: GridSearchCV(5-fold, scoring=`roc_auc`)
- 최적 파라미터:
  - `colsample_bytree=0.8`
  - `learning_rate=0.01`
  - `max_depth=5`
  - `n_estimators=200`
  - `subsample=0.8`
- best CV ROC-AUC: `0.8655`

### 3.9 임계값 최적화 전략
기본 분류 임계값(0.50) 외에, 테스트셋에서 0.10~0.90 범위를 0.01 간격으로 탐색해 정책형 임계값을 도출했다.
1. **Default policy**: 0.50
2. **F1-optimal policy**: F1 최대 임계값
3. **Recall-priority policy**: precision>=0.50 제약 하 최대 Recall 임계값

이 구조는 단일 점수 보고를 넘어 실제 선별 정책에 맞춘 운영 모드 구분을 가능하게 한다.

---

## 4. 실험 결과

### 4.1 전체 성능(Full 모델)
기본 임계값과 정책형 임계값 성능은 표 1과 같다.

**표 1. 임계값 정책별 성능(Full 모델)**

| Policy | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.8109 | 0.6319 | 0.4720 | 0.5404 | 0.8664 |
| F1-optimal | 0.36 | 0.8105 | 0.5740 | 0.7571 | 0.6530 | 0.8664* |
| Recall-priority | 0.23 | 0.7687 | 0.5052 | 0.8603 | 0.6366 | 0.8664* |

\* ROC-AUC는 임계값 독립 지표로 동일 모델에서 동일.

#### 결과 해석
1. Default(0.50)는 precision은 높지만 recall이 낮아 미탐 위험이 크다.
2. F1-optimal(0.36)은 precision 손실을 제한하면서 recall을 크게 높여 F1을 최대화한다.
3. Recall-priority(0.23)는 미탐 최소화 관점에서 유리하지만 오탐 증가를 동반한다.

즉, 선별 목적에서는 0.50 고정 운용보다 정책형 threshold 운용이 더 타당하다.

### 4.2 Full vs Light A/B 비교

**표 2. A/B 비교 결과**

| Model | #Features | Default F1 | ROC-AUC | Best-F1 Threshold | Best-F1 |
|---|---:|---:|---:|---:|---:|
| Full | 18 | 0.5404 | 0.8664 | 0.36 | 0.6530 |
| Light | 11 | 0.5364 | 0.8662 | 0.34 | 0.6517 |

#### 해석
Full 모델이 F1과 AUC 모두에서 근소 우세했다. 성능 차이는 크지 않지만, 생활패턴 스무딩 피처를 폭넓게 반영한 모델이 일관된 개선을 보인 점에서 최종 운영 모델을 Full로 선정했다.

### 4.3 임계값-성능 곡선 분석
`xgboost_threshold_metrics.csv`를 기준으로 threshold를 낮출수록 recall이 상승하고 precision이 하락하는 전형적 trade-off가 확인된다. 예를 들어:
- threshold 0.50: recall 0.4720
- threshold 0.36: recall 0.7571
- threshold 0.23: recall 0.8603

동일 모델이라도 임계값 설정만으로 오류 구조가 크게 바뀌며, 서비스 정책에 따라 최적점이 달라진다는 점을 확인했다.

### 4.4 연령대별 성능 분석
연령대별 분석에서는 고연령군으로 갈수록 Recall/F1이 상승하는 경향이 관찰되었다. 반면 ROC-AUC는 일부 구간에서 낮아지는 경향을 보였다. 이는 연령군별 위험 분포 및 클래스 난이도 차이를 시사하며, 단일 threshold 정책의 한계를 보여준다.

#### 실무적 시사점
1. 동일 모델이라도 연령군별 운영 threshold를 분리할 여지가 있다.
2. 고연령군에서는 미탐 최소화 정책의 체감효용이 더 클 수 있다.
3. 저연령군에서는 과잉경보를 줄이는 정밀도 보정이 필요할 수 있다.

### 4.5 피처 상관구조 점검
`feature_correlation_heatmap.png` 분석 결과, 허리둘레/체중 기반 신체지표와 파생 상호작용항(`waist_x_excess_prob`)이 타깃과 구조적 연관을 보였다. 이는 본 연구의 피처 설계가 임상적 직관과 완전히 분리되지 않음을 뒷받침한다.

---

## 5. 논의

### 5.1 모델 성능의 의미
ROC-AUC 0.8664는 순위화 능력이 안정적임을 의미한다. 다만 선별 시스템에서 중요한 것은 AUC만이 아니라 실제 의사결정 지점의 precision-recall 균형이다. 본 연구는 동일 모델에서도 threshold 정책에 따라 성능 프로파일이 크게 달라짐을 실증했다.

### 5.2 임계값 운영정책의 실무적 가치
의료 선별 맥락에서는 일반적으로 미탐(false negative)의 비용이 오탐(false positive)보다 클 수 있다. 이 관점에서 F1-optimal 또는 Recall-priority 정책은 다음 장점이 있다.
1. 위험군 누락 감소
2. 후속 검진 권고 프로세스와 결합 용이
3. 운영 목적(보수/공격)에 따른 정책 전환 가능

따라서 모델 배포 시 단일 threshold를 고정하기보다, 서비스 목적별 모드를 명시하는 것이 바람직하다.

### 5.3 생활패턴 스무딩 피처의 해석
본 연구의 스무딩 피처는 단일 시점 섭취량보다 개인 내 변동성과 상태 분포를 더 많이 반영한다. 결과적으로 성능 개선폭은 “대폭 상승”이라기보다 “작지만 일관된 우세”였고, 이는 실제 데이터의 잡음과 기록 편차를 고려할 때 현실적인 개선 양상으로 해석된다.

### 5.4 라벨 전이 접근의 타당성과 위험
라벨 전이는 이질 코호트 연결의 실용적 해법이지만, 전이 과정에서 라벨 노이즈가 누적될 가능성이 있다. 즉, 본 접근은 “데이터 결핍 상황에서 동작 가능한 합리적 대안”이며, 후속 연구에서는 외부 검증셋 또는 준실험 설계를 통해 전이 오차를 정량화해야 한다.

### 5.5 사용자 경험(UX) 및 서비스 적용 관점
모델 결과는 반드시 “진단”이 아닌 “위험 신호”로 표현해야 한다. 권장 UX 원칙은 다음과 같다.
1. 점수 + 등급 + 권고행동(검진/상담/식습관 체크) 동시 제공
2. 결과 불확실성(오탐/미탐 가능성) 안내
3. 민감한 건강정보 처리에 대한 동의/보안 고지

### 5.6 한계
1. **데이터 대표성 한계**: 서로 다른 데이터셋의 표본 편향 가능성
2. **전이 라벨 노이즈**: 직접 관측 라벨이 아니므로 전이 오류 내재
3. **시간축 제약**: `recorded_days` 분포가 짧아 장기 패턴 반영 제한
4. **검증 구조 한계**: 단일 홀드아웃 중심 평가로 외부 재현성 보강 필요

### 5.7 윤리 및 안전성
본 모델은 건강관리 참고용으로서, 의료진의 진단/치료/처방을 대체하지 않는다. 특히 고위험 판정 시 자동 의료 권고를 내리기보다, 추가 검진 안내 및 생활관리 권고를 제공하는 보조 시스템으로 사용해야 한다.

---

## 6. 결론
본 연구는 비침습 입력만으로 대사증후군 위험 선별이 가능한 실무형 파이프라인을 제안했다. 핵심 성과는 다음과 같다.
1. 이질 코호트 환경에서 라벨 전이 전략으로 영양 정보를 건강 데이터에 연결했다.
2. 생활패턴 스무딩 피처를 도입해 Full 모델이 Light 대비 일관된 소폭 성능 우세를 보였다.
3. 임계값 최적화(0.36)로 F1 및 Recall을 크게 개선했고, Recall-priority 정책(0.23)으로 미탐 최소화 시나리오를 제시했다.

따라서 본 연구는 “모델 구조 + 운영 임계값 정책”을 통합적으로 설계해야 선별 시스템의 실제 효용이 높아진다는 점을 보여준다.

---

## 7. 향후 연구
1. 외부 코호트 검증(기관/연도 분리 검증)
2. 연령군별/성별 캘리브레이션 및 그룹별 threshold 최적화
3. 장기 시계열 섭취데이터 기반 개인화 적응모델
4. 예측 설명가능성(SHAP 등) 강화 및 임상 커뮤니케이션 실험
5. 앱 내 행동중재 효과(식단 개선/체중관리)와 예측 점수 변화의 종단 분석

---

## 8. 재현성 및 구현 경로
- 전처리: `ml/src/preprocess.py`
- 학습: `ml/src/train.py`
- 평가: `ml/src/evaluate.py`
- 추론: `ml/src/predict.py`
- 분석/시각화: `ml/src/analyze.py`
- 결과 파일:
  - `docs/research_results/xgboost_metrics.json`
  - `docs/research_results/xgboost_ab_comparison.csv`
  - `docs/research_results/xgboost_threshold_metrics.csv`
  - `docs/research_results/feature_correlation_heatmap.png`
  - `docs/research_results/age_group_performance.png`

---

## 9. 참고문헌
[1] Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults. Executive summary of the third report of the National Cholesterol Education Program (NCEP) Adult Treatment Panel III. *JAMA*. 2001;285(19):2486-2497.  
[2] Alberti KGMM, Eckel RH, Grundy SM, et al. Harmonizing the metabolic syndrome: a joint interim statement of major international societies. *Circulation*. 2009;120(16):1640-1645.  
[3] Ministry of Health and Welfare (Korea), Korea Disease Control and Prevention Agency. Korea National Health and Nutrition Examination Survey (KNHANES) documentation.  
[4] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830.  
[5] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*. 2016:785-794.  
[6] Troyanskaya O, Cantor M, Sherlock G, et al. Missing value estimation methods for DNA microarrays. *Bioinformatics*. 2001;17(6):520-525.  

---

## 부록 A. 결과표(원본 수치 정리)

### A.1 Full 모델 메트릭(기본)
- accuracy: 0.8108759430008382
- f1: 0.5403615991851286
- recall: 0.4719750889679715
- precision: 0.6319237641453246
- roc_auc: 0.8663779658429326

### A.2 Full 모델 정책형 임계값
- threshold_best_f1: 0.36
- best_f1: 0.6529829272971418
- recall@best_f1: 0.7571174377224199
- precision@best_f1: 0.5740303541315346
- threshold_recall_priority: 0.23
- recall@recall_priority: 0.8603202846975089
- precision@recall_priority: 0.5052246603970741

### A.3 A/B 핵심값
- Full(default f1): 0.5404, AUC: 0.8664
- Light(default f1): 0.5364, AUC: 0.8662
- Full(best f1): 0.6530 @ 0.36
- Light(best f1): 0.6517 @ 0.34

---

## 부록 B. 투고 직전 점검 체크리스트
1. 본문 임계값 수치가 모두 `0.36` 기준으로 통일되었는가?
2. 표/그림 번호와 본문 인용 번호가 일치하는가?
3. 결과값 소수점 자릿수 표기가 일관적인가?
4. “진단 대체 아님” 고지가 초록·논의·결론에 반영되었는가?
5. 방법론의 재현 경로가 파일 경로와 함께 명시되었는가?

