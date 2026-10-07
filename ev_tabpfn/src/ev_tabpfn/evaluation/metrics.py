from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from .labels import ClassificationLabelContract

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


def _safe_log_loss(y_true: Any, y_prob: np.ndarray | None, labels: list[int] | None = None) -> float | None:
    if y_prob is None:
        return None
    try:
        return float(log_loss(y_true, y_prob, labels=labels))
    except Exception:
        return None


def _safe_roc_auc(
    task_type: str,
    y_true: Any,
    y_prob: np.ndarray | None,
    positive_index: int | None = None,
    labels: list[int] | None = None,
) -> float | None:
    if y_prob is None:
        return None
    try:
        if task_type == "binary":
            pos_index = 1 if positive_index is None else positive_index
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                return float(roc_auc_score(y_true, y_prob[:, pos_index]))
            return float(roc_auc_score(y_true, y_prob))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels))
    except Exception:
        return None


def classification_metrics(
    task_type: str,
    y_true: Any,
    y_pred: Any,
    y_prob: np.ndarray | None,
    *,
    label_contract: ClassificationLabelContract | None = None,
    y_prob_classes: Any | None = None,
) -> dict[str, float | None]:
    encoded_labels = None
    positive_index = None
    if label_contract is not None:
        y_true = label_contract.encode(y_true)
        y_pred = label_contract.encode(y_pred)
        y_prob = label_contract.align_probabilities(y_prob, y_prob_classes)
        encoded_labels = label_contract.encoded_classes
        positive_index = label_contract.positive_index

    f1_average = "binary" if task_type == "binary" else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=f1_average)),
        "roc_auc": _safe_roc_auc(task_type, y_true, y_prob, positive_index, encoded_labels),
        "log_loss": _safe_log_loss(y_true, y_prob, encoded_labels),
    }


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | None]:
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if spearmanr is None:
        metrics["spearman"] = None
    else:
        try:
            metrics["spearman"] = float(spearmanr(y_true, y_pred).statistic)
        except Exception:
            metrics["spearman"] = None
    return metrics

