from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from .autogluon_adapter import AutoGluonAdapter
except Exception:  # pragma: no cover - optional dependency
    AutoGluonAdapter = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task_type: str
    family: str
    estimator: Any
    requires_encoded_target: bool = False


def _build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ]
    if scale_numeric:
        categorical_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_exclude=["object", "category", "bool"])),
            ("cat", Pipeline(categorical_steps), make_column_selector(dtype_include=["object", "category", "bool"])),
        ],
        remainder="drop",
    )


def make_column_selector(*, dtype_include=None, dtype_exclude=None):
    from sklearn.compose import make_column_selector as _selector

    return _selector(dtype_include=dtype_include, dtype_exclude=dtype_exclude)


def _tabpfn_classifier() -> Any:
    from tabpfn import TabPFNClassifier
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNClassifier(device=device)


def _tabpfn_regressor() -> Any:
    from tabpfn import TabPFNRegressor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNRegressor(device=device)


def _baseline_pipeline(*, scale_numeric: bool, model: Any) -> Pipeline:
    return Pipeline([("prep", _build_preprocessor(scale_numeric=scale_numeric)), ("model", model)])


def _autogluon_estimator(
    *,
    task_type: str,
    run_dir: str | Path | None,
    autogluon_config: dict[str, Any] | None,
) -> Any | None:
    if AutoGluonAdapter is None or run_dir is None:
        return None

    config = autogluon_config or {}
    model_path = Path(run_dir) / "autogluon_models" / task_type
    return AutoGluonAdapter(
        problem_type=task_type,
        model_path=model_path,
        presets=config.get("presets", "medium_quality"),
        time_limit=config.get("time_limit", 60.0),
        eval_metric=config.get("eval_metric"),
        verbosity=config.get("verbosity", 0),
    )


def _append_optional_classifier_models(specs: list[ModelSpec], task_type: str, num_classes: int | None) -> None:
    if CatBoostClassifier is not None:
        specs.append(
            ModelSpec(
                "catboost",
                task_type,
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=CatBoostClassifier(
                        random_state=42,
                        verbose=0,
                    ),
                ),
            )
        )

    if XGBClassifier is not None:
        xgb_kwargs = {
            "random_state": 42,
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
        if task_type == "binary":
            xgb_kwargs["objective"] = "binary:logistic"
            xgb_kwargs["eval_metric"] = "logloss"
        else:
            xgb_kwargs["objective"] = "multi:softprob"
            xgb_kwargs["eval_metric"] = "mlogloss"
            if num_classes is not None:
                xgb_kwargs["num_class"] = num_classes
        specs.append(
            ModelSpec(
                "xgboost",
                task_type,
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=XGBClassifier(**xgb_kwargs),
                ),
                requires_encoded_target=True,
            )
        )

    if LGBMClassifier is not None:
        objective = "binary" if task_type == "binary" else "multiclass"
        specs.append(
            ModelSpec(
                "lightgbm",
                task_type,
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=LGBMClassifier(
                        random_state=42,
                        n_estimators=300,
                        learning_rate=0.05,
                        objective=objective,
                        verbose=-1,
                    ),
                ),
            )
        )


def _append_optional_autogluon_classifier(
    specs: list[ModelSpec],
    *,
    task_type: str,
    run_dir: str | Path | None,
    autogluon_config: dict[str, Any] | None,
) -> None:
    estimator = _autogluon_estimator(
        task_type=task_type,
        run_dir=run_dir,
        autogluon_config=autogluon_config,
    )
    if estimator is not None:
        specs.append(ModelSpec("autogluon", task_type, "baseline", estimator))


def _append_optional_regressor_models(specs: list[ModelSpec], task_type: str) -> None:
    if CatBoostRegressor is not None:
        specs.append(
            ModelSpec(
                "catboost",
                task_type,
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=CatBoostRegressor(
                        random_state=42,
                        verbose=0,
                    ),
                ),
            )
        )

    if XGBRegressor is not None:
        specs.append(
            ModelSpec(
                "xgboost",
                task_type,
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=XGBRegressor(
                        random_state=42,
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                    ),
                ),
            )
        )

    if LGBMRegressor is not None:
        specs.append(
            ModelSpec(
                "lightgbm",
                task_type,
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=LGBMRegressor(
                        random_state=42,
                        n_estimators=300,
                        learning_rate=0.05,
                        verbose=-1,
                    ),
                ),
            )
        )


def _append_optional_autogluon_regressor(
    specs: list[ModelSpec],
    *,
    task_type: str,
    run_dir: str | Path | None,
    autogluon_config: dict[str, Any] | None,
) -> None:
    estimator = _autogluon_estimator(
        task_type=task_type,
        run_dir=run_dir,
        autogluon_config=autogluon_config,
    )
    if estimator is not None:
        specs.append(ModelSpec("autogluon", task_type, "baseline", estimator))


def build_models(
    task_type: str,
    y_train: Any | None = None,
    *,
    run_dir: str | Path | None = None,
    autogluon_config: dict[str, Any] | None = None,
) -> list[ModelSpec]:
    num_classes = None
    if task_type == "multiclass" and y_train is not None:
        num_classes = int(len(np.unique(np.asarray(y_train))))

    if task_type == "binary":
        specs = [
            ModelSpec("tabpfn", "binary", "tabpfn", _tabpfn_classifier()),
            ModelSpec(
                "random_forest",
                "binary",
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ),
            ModelSpec(
                "logistic_regression",
                "binary",
                "baseline",
                _baseline_pipeline(
                    scale_numeric=True,
                    model=LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
            ),
        ]
        _append_optional_classifier_models(specs, task_type, num_classes=None)
        _append_optional_autogluon_classifier(
            specs,
            task_type=task_type,
            run_dir=run_dir,
            autogluon_config=autogluon_config,
        )
        return specs

    if task_type == "multiclass":
        specs = [
            ModelSpec("tabpfn", "multiclass", "tabpfn", _tabpfn_classifier()),
            ModelSpec(
                "random_forest",
                "multiclass",
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ),
            ModelSpec(
                "logistic_regression",
                "multiclass",
                "baseline",
                _baseline_pipeline(
                    scale_numeric=True,
                    model=LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        multi_class="auto",
                    ),
                ),
            ),
        ]
        _append_optional_classifier_models(specs, task_type, num_classes=num_classes)
        _append_optional_autogluon_classifier(
            specs,
            task_type=task_type,
            run_dir=run_dir,
            autogluon_config=autogluon_config,
        )
        return specs

    if task_type == "regression":
        specs = [
            ModelSpec("tabpfn", "regression", "tabpfn", _tabpfn_regressor()),
            ModelSpec(
                "random_forest",
                "regression",
                "baseline",
                _baseline_pipeline(
                    scale_numeric=False,
                    model=RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ),
            ModelSpec(
                "ridge",
                "regression",
                "baseline",
                _baseline_pipeline(
                    scale_numeric=True,
                    model=Ridge(alpha=1.0, random_state=42),
                ),
            ),
        ]
        _append_optional_regressor_models(specs, task_type)
        _append_optional_autogluon_regressor(
            specs,
            task_type=task_type,
            run_dir=run_dir,
            autogluon_config=autogluon_config,
        )
        return specs

    raise ValueError(f"Unsupported task_type: {task_type}")
