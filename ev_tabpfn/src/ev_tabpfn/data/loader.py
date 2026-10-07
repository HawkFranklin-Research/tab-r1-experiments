from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetContainer:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    task_type: str
    feature_names: list[str]
    target_name: str
    metadata: dict[str, Any]


class DataLoader:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.logger = logging.getLogger("ev_tabpfn.data")

    def infer_task(self, y: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.dropna().unique()
            if len(unique_vals) <= 2:
                return "binary"
            if len(unique_vals) <= 10 and all(float(val).is_integer() for val in unique_vals):
                return "multiclass"
            return "regression"
        unique_vals = y.dropna().unique()
        return "binary" if len(unique_vals) == 2 else "multiclass"

    def load_local_csv(
        self,
        file_path: str,
        target_column: str | None = None,
        val_size: float = 0.15,
        test_size: float = 0.15,
        task_override: str | None = None,
    ) -> DatasetContainer:
        df = pd.read_csv(file_path, sep=None, engine="python")
        if target_column is None:
            target_column = str(df.columns[-1])
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        task_type = task_override or self.infer_task(y)
        if task_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(f"Unsupported task type: {task_type}")

        stratify = y if task_type in {"binary", "multiclass"} else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.seed,
            stratify=stratify,
        )

        relative_val_size = val_size / (1 - test_size)
        stratify_temp = y_temp if task_type in {"binary", "multiclass"} else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=self.seed,
            stratify=stratify_temp,
        )

        return DatasetContainer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
            feature_names=X.columns.tolist(),
            target_name=target_column,
            metadata={
                "total_samples": len(df),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "n_features": X.shape[1],
                "file_source": file_path,
            },
        )

