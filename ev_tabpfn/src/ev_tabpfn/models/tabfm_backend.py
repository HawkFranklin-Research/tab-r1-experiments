from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TabFMAdapter:
    task_type: str
    backend: str = "jax"
    ensemble: bool = False
    checkpoint_path: str | None = None
    max_classes: int = 10
    max_train_rows: int | None = None
    random_state: int = 42
    load_kwargs: dict[str, Any] | None = None
    estimator_kwargs: dict[str, Any] | None = None

    def _load_model(self) -> Any:
        model_type = "regression" if self.task_type == "regression" else "classification"
        kwargs = dict(self.load_kwargs or {})
        if self.checkpoint_path:
            kwargs.setdefault("checkpoint_path", self.checkpoint_path)

        if self.backend == "jax":
            from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
        elif self.backend == "pytorch":
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        else:
            raise ValueError(f"Unsupported TabFM backend: {self.backend!r}. Expected 'jax' or 'pytorch'.")

        return tabfm_v1_0_0.load(model_type=model_type, **kwargs)

    def _build_estimator(self) -> Any:
        kwargs = {"random_state": self.random_state, **dict(self.estimator_kwargs or {})}
        if not self.ensemble:
            kwargs.setdefault("n_estimators", 1)
            kwargs.setdefault("batch_size", 1)
            kwargs.setdefault("num_folds_for_cv", 2)
        model = self._load_model()
        if self.task_type == "regression":
            from tabfm import TabFMRegressor

            return TabFMRegressor.ensemble(model, **kwargs) if self.ensemble else TabFMRegressor(model, **kwargs)

        from tabfm import TabFMClassifier

        return TabFMClassifier.ensemble(model, **kwargs) if self.ensemble else TabFMClassifier(model, **kwargs)

    def _maybe_cap_train_rows(self, X: Any, y: Any) -> tuple[Any, Any]:
        if self.max_train_rows is None:
            return X, y
        from ev_tabpfn.evaluation.generation_preprocessing import cap_training_rows

        capped = cap_training_rows(
            X,
            y,
            task_type=self.task_type,
            seed=self.random_state,
            max_rows=self.max_train_rows,
        )
        return capped.X_train, capped.y_train

    def fit(self, X: Any, y: Any) -> "TabFMAdapter":
        if self.task_type in {"binary", "multiclass"}:
            n_classes = len(np.unique(np.asarray(y)))
            if n_classes > self.max_classes:
                raise ValueError(f"TabFM supports at most {self.max_classes} classes, got {n_classes}.")

        X_fit, y_fit = self._maybe_cap_train_rows(X, y)
        self.estimator_ = self._build_estimator()
        self.estimator_.fit(X_fit, y_fit)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        self._cached_proba_input_id = None
        self._cached_proba = None
        return self

    def predict(self, X: Any) -> Any:
        if self.task_type in {"binary", "multiclass"}:
            y_prob = self.predict_proba(X)
            classes = np.asarray(self.classes_)
            return classes[np.asarray(y_prob).argmax(axis=1)]
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("TabFM regressor does not expose predict_proba.")
        input_id = id(X)
        if self._cached_proba_input_id == input_id and self._cached_proba is not None:
            return self._cached_proba
        self._cached_proba = self.estimator_.predict_proba(X)
        self._cached_proba_input_id = input_id
        return self._cached_proba
