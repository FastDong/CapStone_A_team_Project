# ML Project

이 폴더는 대사증후군 예측 모델 학습 프로젝트입니다.

## Main Tasks

- KNHANES 데이터 전처리
- 학습용 데이터셋 생성
- 모델 학습 및 평가
- 추론 모델 export

## Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

## Input

- 원본 데이터는 `../data/raw/`에 위치해야 합니다.

## Output

- 전처리 결과: `../data/processed/`
- 학습 모델: `./models/`
- 평가 결과: `./outputs/`
