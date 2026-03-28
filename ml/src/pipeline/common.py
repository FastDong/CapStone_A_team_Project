from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_INTERIM_DIR = ROOT_DIR / "data" / "interim"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "ml" / "models"
OUTPUT_DIR = ROOT_DIR / "ml" / "outputs"


def ensure_dirs() -> None:
    for path in [DATA_INTERIM_DIR, DATA_PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_id_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r'^b"(.*)"$', r"\1", regex=True)
    s = s.str.replace(r"^b'(.*)'$", r"\1", regex=True)
    s = s.replace({"nan": np.nan, "None": np.nan})
    return s


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@dataclass
class SoftmaxKNNOutput:
    labels: np.ndarray
    probs: np.ndarray


def knn_predict_with_softmax(
    knn: KNeighborsClassifier,
    x: np.ndarray,
    class_values: np.ndarray,
    temperature: float = 1.0,
) -> SoftmaxKNNOutput:
    distances, indices = knn.kneighbors(x, return_distance=True)
    neighbor_labels = knn._y[indices]  # sklearn stores fitted labels here.
    weights = 1.0 / (distances + 1e-6)

    scores = np.zeros((x.shape[0], len(class_values)), dtype=np.float64)
    class_to_idx = {c: i for i, c in enumerate(class_values)}
    for row_idx in range(neighbor_labels.shape[0]):
        for n_idx in range(neighbor_labels.shape[1]):
            cls = neighbor_labels[row_idx, n_idx]
            scores[row_idx, class_to_idx[cls]] += weights[row_idx, n_idx]

    probs = softmax(scores / max(temperature, 1e-6), axis=1)
    labels = class_values[np.argmax(probs, axis=1)]
    return SoftmaxKNNOutput(labels=labels, probs=probs)
