# Metabolic Syndrome Capstone Project

이 저장소는 **대사증후군 예측 시스템**을 위한 캡스톤 프로젝트 리포지토리입니다.
서로 연결된 두 개의 프로젝트로 구성되어 있습니다.

1. **대사증후군 예측 학습 프로젝트 (`ml/`)**
   - KNHANES 2017-2023 데이터를 기반으로
   - 데이터 전처리, 특성 선택, 모델 학습 및 평가를 수행합니다.

2. **대사증후군 예측 모델을 활용한 앱 프로젝트 (`app/`)**
   - 학습된 예측 모델을 활용하여
   - 사용자가 건강 정보를 입력하면 예측 결과를 확인할 수 있는 앱을 개발합니다.

---

## Repository Structure

```text
.
├─ README.md
├─ .gitignore
├─ LICENSE
├─ docs/                  # 문서 및 설계 자료
├─ data/
│  ├─ raw/                # KNHANES 원본 데이터 위치 (git 추적 안 함)
│  ├─ interim/            # 전처리 중간 데이터 (git 추적 안 함)
│  └─ processed/          # 최종 학습용 데이터 (git 추적 안 함)
├─ ml/                    # 머신러닝 학습 프로젝트
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ src/
│  ├─ notebooks/
│  ├─ models/             # 학습 모델 저장 폴더 (git 추적 안 함)
│  └─ outputs/            # 결과 파일 저장 폴더 (git 추적 안 함)
├─ app/                   # 앱 프로젝트
│  ├─ README.md
│  ├─ package.json
│  ├─ src/
│  ├─ assets/
│  └─ .env.example
└─ scripts/               # 실행 및 보조 스크립트
```

---

## Project Goals

- KNHANES 데이터를 활용한 대사증후군 예측 모델 개발
- 예측 성능이 확보된 모델을 앱에 연결
- 사용자가 쉽게 건강 위험도를 확인할 수 있는 서비스 프로토타입 구현

---

## Data Policy

본 프로젝트는 **KNHANES 2017-2023** 데이터를 사용합니다.

### Important Notice

- KNHANES 원본 데이터 및 가공된 개인 단위 데이터는 이 저장소에 포함하지 않습니다.
- 원본 데이터는 공식 경로를 통해 개별적으로 확보해야 합니다.
- 사용자는 데이터를 `data/raw/` 경로에 직접 배치한 뒤 전처리 및 학습 스크립트를 실행해야 합니다.
- 이 저장소에는 코드, 문서, 실행 예시, 더미 샘플만 포함하는 것을 원칙으로 합니다.

### Expected Data Paths

```text
data/raw/
data/interim/
data/processed/
```

---

## Components

### 1) ML Project (`ml/`)

주요 기능:

- 데이터 정제 및 결측치 처리
- 변수 선택 및 라벨 정의
- 모델 학습
- 성능 평가
- 추론용 모델 export

예상 파일:

- `src/preprocess.py`
- `src/train.py`
- `src/evaluate.py`
- `src/predict.py`

### 2) App Project (`app/`)

주요 기능:

- 사용자 입력 수집
- 예측 모델 결과 표시
- 건강 관련 안내 UI 제공
- 필요 시 API 또는 내장 모델 연동

---

## Getting Started

### ML project

```bash
cd ml
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

### App project

```bash
cd app
npm install
npm run start
```

---

## Environment Variables

민감한 설정값은 `.env` 파일에 저장하고 Git에 커밋하지 않습니다.
예시는 `app/.env.example` 파일을 참고하세요.

예시:

```env
API_BASE_URL=
MODEL_ENDPOINT=
```

---

## Development Rules

이 저장소는 AI 코딩 에이전트(Codex 등)가 구조를 쉽게 이해하고 관리할 수 있도록 다음 원칙을 따릅니다.

- `ml/`와 `app/`의 역할을 명확히 분리
- 데이터 원본 및 대용량 산출물은 Git에 포함하지 않음
- 실행 방법은 각 폴더의 `README.md`에 명시
- 문서는 `docs/`에 정리
- 기능 단위 브랜치 사용 권장

권장 브랜치 예시:

- `feature/ml-preprocessing`
- `feature/ml-training`
- `feature/app-ui`
- `feature/app-model-integration`
- `docs/readme-update`

---

## Disclaimer

이 프로젝트는 **연구 및 교육 목적의 캡스톤 프로젝트**입니다.
앱 또는 예측 결과는 **의료적 진단, 치료, 처방을 대체하지 않습니다.**
실제 의료 판단은 반드시 의료 전문가와 상담해야 합니다.

---

## License

This repository uses the **Apache License 2.0**.
