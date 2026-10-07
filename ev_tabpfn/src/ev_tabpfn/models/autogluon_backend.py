from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


class AutoGluonAdapter:
    def __init__(
        self,
        *,
        problem_type: str,
        model_path: str | Path,
        presets: str | None = "medium_quality",
        time_limit: float | None = 60.0,
        eval_metric: str | None = None,
        verbosity: int = 0,
    ) -> None:
        self.problem_type = problem_type
        self.model_path = Path(model_path)
        self.presets = presets
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.verbosity = verbosity
        self.label_column = "__target__"
        self.predictor: TabularPredictor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoGluonAdapter":
        train_data = X.reset_index(drop=True).copy()
        train_data[self.label_column] = pd.Series(y).reset_index(drop=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.predictor = TabularPredictor(
            label=self.label_column,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=str(self.model_path),
            verbosity=self.verbosity,
            log_to_file=False,
        )
        self.predictor.fit(train_data=train_data, time_limit=self.time_limit, presets=self.presets)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_fit()
        predictions = self.predictor.predict(X.reset_index(drop=True))
        return np.asarray(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_fit()
        probabilities = self.predictor.predict_proba(
            X.reset_index(drop=True),
            as_pandas=True,
            as_multiclass=True,
        )
        if isinstance(probabilities, pd.Series):
            pos = probabilities.to_numpy()
            return np.column_stack([1.0 - pos, pos])
        if isinstance(probabilities, pd.DataFrame):
            return probabilities.to_numpy()
        return np.asarray(probabilities)

    def _ensure_fit(self) -> None:
        if self.predictor is None:
            raise RuntimeError("AutoGluonAdapter must be fit before prediction.")

