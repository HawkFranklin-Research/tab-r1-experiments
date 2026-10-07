from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class TrainingCapResult:
    X_train: pd.DataFrame
    y_train: Any
    rows_used: int
    rows_original: int
    cap: int | None


class TabPFNGenerationPreprocessor:
    def __init__(self) -> None:
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.numeric_medians: pd.Series | None = None
        self.categorical_map: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame) -> "TabPFNGenerationPreprocessor":
        self.numeric_cols = list(X.select_dtypes(exclude=["object", "category", "bool"]).columns)
        self.categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
        if self.numeric_cols:
            self.numeric_medians = X[self.numeric_cols].apply(pd.to_numeric, errors="coerce").median()
        else:
            self.numeric_medians = pd.Series(dtype=float)
        self.categorical_map = {}
        for col in self.categorical_cols:
            values = X[col].astype("string").fillna("__MISSING__")
            self.categorical_map[col] = sorted(values.unique().tolist())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_cols:
            numeric = X[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(self.numeric_medians)
            parts.append(numeric.to_numpy(dtype=np.float32, copy=False))
        if self.categorical_cols:
            for col in self.categorical_cols:
                categories = self.categorical_map.get(col, [])
                mapping = {value: idx for idx, value in enumerate(categories)}
                series = X[col].astype("string").fillna("__MISSING__")
                encoded = series.map(mapping).fillna(-1).to_numpy(dtype=np.float32).reshape(-1, 1)
                parts.append(encoded)
        if not parts:
            return np.empty((len(X), 0), dtype=np.float32)
        return np.hstack(parts)


def cap_training_rows(
    X_train: pd.DataFrame,
    y_train: Any,
    *,
    task_type: str,
    seed: int,
    max_rows: int | None,
) -> TrainingCapResult:
    original_rows = len(X_train)
    if max_rows is None or original_rows <= max_rows:
        return TrainingCapResult(X_train=X_train, y_train=y_train, rows_used=original_rows, rows_original=original_rows, cap=max_rows)

    stratify = y_train if task_type in {"binary", "multiclass"} else None
    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_rows,
        random_state=seed,
        stratify=stratify,
    )
    return TrainingCapResult(X_train=X_subset, y_train=y_subset, rows_used=len(X_subset), rows_original=original_rows, cap=max_rows)

