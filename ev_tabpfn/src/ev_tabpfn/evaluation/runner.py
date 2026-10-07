from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tabpfn.evaluation.labels import ClassificationLabelContract, flatten_predictions
from ev_tabpfn.evaluation.metrics import classification_metrics, regression_metrics
from ev_tabpfn.models.registry import ModelSpec


@dataclass
class ModelRunResult:
    model_name: str
    family: str
    task_type: str
    status: str
    fit_time_s: float | None
    predict_time_s: float | None
    error_type: str | None
    error_message: str | None
    traceback_text: str | None
    metrics: dict[str, float | None]
    y_pred: Any = None
    y_prob: np.ndarray | None = None
    y_prob_classes: list[Any] | None = None

    def to_summary_row(self) -> dict[str, Any]:
        row = {
            "model_name": self.model_name,
            "family": self.family,
            "task_type": self.task_type,
            "status": self.status,
            "fit_time_s": self.fit_time_s,
            "predict_time_s": self.predict_time_s,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
        row.update(self.metrics)
        return row


def _predict_proba_if_available(estimator: Any, X: Any) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        try:
            return estimator.predict_proba(X)
        except Exception:
            return None
    return None


def _classes_if_available(estimator: Any) -> list[Any] | None:
    if hasattr(estimator, "classes_"):
        try:
            return list(estimator.classes_)
        except Exception:
            return None
    if hasattr(estimator, "named_steps"):
        final_step = list(estimator.named_steps.values())[-1]
        if hasattr(final_step, "classes_"):
            try:
                return list(final_step.classes_)
            except Exception:
                return None
    return None


def run_model(
    spec: ModelSpec,
    dataset: Any,
    *,
    label_contract: ClassificationLabelContract | None = None,
) -> ModelRunResult:
    try:
        y_train = dataset.y_train
        if spec.requires_encoded_target:
            if label_contract is None:
                raise ValueError(f"Model {spec.name} requires encoded targets but no label contract was provided.")
            y_train = label_contract.encode(dataset.y_train)

        start_fit = time.perf_counter()
        spec.estimator.fit(dataset.X_train, y_train)
        fit_time = time.perf_counter() - start_fit

        start_predict = time.perf_counter()
        y_pred = spec.estimator.predict(dataset.X_test)
        if spec.requires_encoded_target:
            y_pred = label_contract.decode(y_pred)
        else:
            y_pred = flatten_predictions(y_pred)
        y_prob = (
            _predict_proba_if_available(spec.estimator, dataset.X_test)
            if dataset.task_type in {"binary", "multiclass"}
            else None
        )
        y_prob_classes = _classes_if_available(spec.estimator)
        if spec.requires_encoded_target and label_contract is not None:
            y_prob_classes = label_contract.classes
        predict_time = time.perf_counter() - start_predict

        if dataset.task_type in {"binary", "multiclass"}:
            metrics = classification_metrics(
                dataset.task_type,
                dataset.y_test,
                y_pred,
                y_prob,
                label_contract=label_contract,
                y_prob_classes=y_prob_classes,
            )
        else:
            metrics = regression_metrics(dataset.y_test, y_pred)

        return ModelRunResult(
            model_name=spec.name,
            family=spec.family,
            task_type=spec.task_type,
            status="success",
            fit_time_s=fit_time,
            predict_time_s=predict_time,
            error_type=None,
            error_message=None,
            traceback_text=None,
            metrics=metrics,
            y_pred=y_pred,
            y_prob=y_prob,
            y_prob_classes=y_prob_classes,
        )
    except Exception as exc:
        return ModelRunResult(
            model_name=spec.name,
            family=spec.family,
            task_type=spec.task_type,
            status="failed",
            fit_time_s=None,
            predict_time_s=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
            metrics={},
            y_pred=None,
            y_prob=None,
        )

