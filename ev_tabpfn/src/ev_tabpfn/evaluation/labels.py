from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def flatten_predictions(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    raise ValueError(f"Predictions must be 1-dimensional, got shape {arr.shape}.")


def _column_safe_label(value: Any) -> str:
    text = str(value)
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "class"


def _probability_columns(classes: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    columns: list[str] = []
    for index, label in enumerate(classes):
        base = f"prob_{_column_safe_label(label)}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{index}")
    return columns


@dataclass
class ClassificationLabelContract:
    task_type: str
    encoder: LabelEncoder
    classes: list[Any]
    class_labels: list[str]
    probability_columns: list[str]
    positive_label: Any | None
    positive_index: int | None

    @classmethod
    def from_labels(cls, task_type: str, labels: Any) -> "ClassificationLabelContract":
        if task_type not in {"binary", "multiclass"}:
            raise ValueError(f"Classification label contract does not support task_type={task_type!r}.")
        encoder = LabelEncoder()
        encoder.fit(flatten_predictions(labels))
        classes = list(encoder.classes_)
        positive_index = 1 if task_type == "binary" and len(classes) == 2 else None
        positive_label = classes[positive_index] if positive_index is not None else None
        return cls(
            task_type=task_type,
            encoder=encoder,
            classes=classes,
            class_labels=[str(value) for value in classes],
            probability_columns=_probability_columns(classes),
            positive_label=positive_label,
            positive_index=positive_index,
        )

    @property
    def encoded_classes(self) -> list[int]:
        return list(range(len(self.classes)))

    def encode(self, values: Any) -> np.ndarray:
        return self.encoder.transform(flatten_predictions(values))

    def decode(self, values: Any) -> np.ndarray:
        return self.encoder.inverse_transform(flatten_predictions(values).astype(int))

    def metadata(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "classes": [str(value) for value in self.classes],
            "encoded_classes": self.encoded_classes,
            "positive_label": None if self.positive_label is None else str(self.positive_label),
            "positive_index": self.positive_index,
            "probability_columns": self.probability_columns,
        }

    def align_probabilities(self, y_prob: Any, y_prob_classes: Any | None = None) -> np.ndarray | None:
        if y_prob is None:
            return None
        arr = np.asarray(y_prob)
        if arr.ndim == 1:
            if self.task_type != "binary":
                raise ValueError("1D probability vectors are only valid for binary classification.")
            arr = np.column_stack([1.0 - arr, arr])
        if arr.ndim != 2:
            raise ValueError(f"Probability matrix must be 2-dimensional, got shape {arr.shape}.")
        expected_width = len(self.classes)
        if arr.shape[1] != expected_width:
            raise ValueError(
                f"Probability matrix has {arr.shape[1]} columns but label contract has {expected_width} classes."
            )
        if y_prob_classes is None:
            return arr
        prob_classes = list(flatten_predictions(y_prob_classes))
        if len(prob_classes) != expected_width:
            return arr
        class_to_index = {str(label): index for index, label in enumerate(prob_classes)}
        if all(str(label) in class_to_index for label in self.classes):
            order = [class_to_index[str(label)] for label in self.classes]
            return arr[:, order]
        return arr

    def probability_frame(self, y_prob: Any, y_prob_classes: Any | None = None) -> pd.DataFrame:
        aligned = self.align_probabilities(y_prob, y_prob_classes)
        if aligned is None:
            return pd.DataFrame()
        return pd.DataFrame(aligned, columns=self.probability_columns)

