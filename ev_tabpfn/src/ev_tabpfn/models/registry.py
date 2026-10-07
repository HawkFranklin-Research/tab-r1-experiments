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
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from .autogluon_backend import AutoGluonAdapter
except Exception:  # pragma: no cover
    AutoGluonAdapter = None

try:
    from .tabfm_backend import TabFMAdapter
except Exception:  # pragma: no cover
    TabFMAdapter = None

from .tabpfn_versions import build_tabpfn_classifier, build_tabpfn_regressor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task_type: str
    family: str
    estimator: Any
    requires_encoded_target: bool = False


def _model_enabled(models: dict[str, dict[str, Any]] | None, name: str, *, default: bool = True) -> bool:
    if not models:
        return default
    return bool(models.get(name, {}).get("enabled", False))


def _build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    from sklearn.compose import make_column_selector

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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


def _baseline_pipeline(*, scale_numeric: bool, model: Any) -> Pipeline:
    return Pipeline([("prep", _build_preprocessor(scale_numeric=scale_numeric)), ("model", model)])


def _autogluon_estimator(
    *,
    task_type: str,
    run_dir: str | Path | None,
    model_config: dict[str, Any] | None,
) -> Any | None:
    if AutoGluonAdapter is None or run_dir is None:
        return None
    config = model_config or {}
    model_path = Path(run_dir) / "autogluon_models" / task_type
    return AutoGluonAdapter(
        problem_type=task_type,
        model_path=model_path,
        presets=config.get("presets", "medium_quality"),
        time_limit=config.get("time_limit", 60.0),
        eval_metric=config.get("eval_metric"),
        verbosity=config.get("verbosity", 0),
    )


def _tabfm_estimator(*, task_type: str, model_config: dict[str, Any] | None) -> Any | None:
    if TabFMAdapter is None:
        return None
    config = dict(model_config or {})
    config.pop("enabled", None)
    return TabFMAdapter(
        task_type=task_type,
        backend=config.pop("backend", "jax"),
        ensemble=bool(config.pop("ensemble", False)),
        checkpoint_path=config.pop("checkpoint_path", None),
        max_classes=int(config.pop("max_classes", 10)),
        max_train_rows=config.pop("max_train_rows", None),
        random_state=int(config.pop("random_state", 42)),
        load_kwargs=dict(config.pop("load_kwargs", {})),
        estimator_kwargs=config,
    )


def _tabpfn_model_config(models: dict[str, dict[str, Any]] | None, name: str) -> dict[str, Any]:
    return dict((models or {}).get(name, {}))


def _append_tabpfn_classifier_specs(
    specs: list[ModelSpec],
    task_type: str,
    *,
    models: dict[str, dict[str, Any]] | None,
) -> None:
    if _model_enabled(models, "tabpfn"):
        config = _tabpfn_model_config(models, "tabpfn")
        specs.append(ModelSpec("tabpfn", task_type, "tabpfn", build_tabpfn_classifier(config=config)))
    for version in ("v2", "v2_5", "v2_6", "v3"):
        name = f"tabpfn_{version}"
        if _model_enabled(models, name, default=False):
            config = _tabpfn_model_config(models, name)
            specs.append(ModelSpec(name, task_type, "tabpfn", build_tabpfn_classifier(version, config=config)))


def _append_tabpfn_regressor_specs(
    specs: list[ModelSpec],
    *,
    models: dict[str, dict[str, Any]] | None,
) -> None:
    if _model_enabled(models, "tabpfn"):
        config = _tabpfn_model_config(models, "tabpfn")
        specs.append(ModelSpec("tabpfn", "regression", "tabpfn", build_tabpfn_regressor(config=config)))
    for version in ("v2", "v2_5", "v2_6", "v3"):
        name = f"tabpfn_{version}"
        if _model_enabled(models, name, default=False):
            config = _tabpfn_model_config(models, name)
            specs.append(ModelSpec(name, "regression", "tabpfn", build_tabpfn_regressor(version, config=config)))


def build_models(
    task_type: str,
    y_train: Any | None = None,
    *,
    run_dir: str | Path | None = None,
    models: dict[str, dict[str, Any]] | None = None,
) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    num_classes = int(len(np.unique(np.asarray(y_train)))) if task_type == "multiclass" and y_train is not None else None

    if task_type in {"binary", "multiclass"}:
        _append_tabpfn_classifier_specs(specs, task_type, models=models)
        if _model_enabled(models, "random_forest"):
            specs.append(
                ModelSpec(
                    "random_forest",
                    task_type,
                    "baseline",
                    _baseline_pipeline(
                        scale_numeric=False,
                        model=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                    ),
                )
            )
        if _model_enabled(models, "logistic_regression"):
            specs.append(
                ModelSpec(
                    "logistic_regression",
                    task_type,
                    "baseline",
                    _baseline_pipeline(
                        scale_numeric=True,
                        model=LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto"),
                    ),
                )
            )
        if CatBoostClassifier is not None and _model_enabled(models, "catboost"):
            specs.append(
                ModelSpec(
                    "catboost",
                    task_type,
                    "baseline",
                    _baseline_pipeline(scale_numeric=False, model=CatBoostClassifier(random_state=42, verbose=0)),
                )
            )
        if XGBClassifier is not None and _model_enabled(models, "xgboost"):
            xgb_kwargs = {
                "random_state": 42,
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "binary:logistic" if task_type == "binary" else "multi:softprob",
                "eval_metric": "logloss" if task_type == "binary" else "mlogloss",
            }
            if task_type == "multiclass" and num_classes is not None:
                xgb_kwargs["num_class"] = num_classes
            specs.append(
                ModelSpec(
                    "xgboost",
                    task_type,
                    "baseline",
                    _baseline_pipeline(scale_numeric=False, model=XGBClassifier(**xgb_kwargs)),
                    requires_encoded_target=True,
                )
            )
        if LGBMClassifier is not None and _model_enabled(models, "lightgbm"):
            objective = "binary" if task_type == "binary" else "multiclass"
            specs.append(
                ModelSpec(
                    "lightgbm",
                    task_type,
                    "baseline",
                    _baseline_pipeline(
                        scale_numeric=False,
                        model=LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.05, objective=objective, verbose=-1),
                    ),
                )
            )
        if _model_enabled(models, "autogluon"):
            estimator = _autogluon_estimator(
                task_type=task_type,
                run_dir=run_dir,
                model_config=(models or {}).get("autogluon"),
            )
            if estimator is not None:
                specs.append(ModelSpec("autogluon", task_type, "baseline", estimator))
        if _model_enabled(models, "tabfm"):
            estimator = _tabfm_estimator(task_type=task_type, model_config=(models or {}).get("tabfm"))
            if estimator is not None:
                specs.append(ModelSpec("tabfm", task_type, "tabfm", estimator))
        return specs

    if task_type == "regression":
        _append_tabpfn_regressor_specs(specs, models=models)
        if _model_enabled(models, "random_forest"):
            specs.append(
                ModelSpec(
                    "random_forest",
                    "regression",
                    "baseline",
                    _baseline_pipeline(
                        scale_numeric=False,
                        model=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                    ),
                )
            )
        if _model_enabled(models, "ridge"):
            specs.append(
                ModelSpec(
                    "ridge",
                    "regression",
                    "baseline",
                    _baseline_pipeline(scale_numeric=True, model=Ridge(alpha=1.0, random_state=42)),
                )
            )
        if CatBoostRegressor is not None and _model_enabled(models, "catboost"):
            specs.append(
                ModelSpec(
                    "catboost",
                    "regression",
                    "baseline",
                    _baseline_pipeline(scale_numeric=False, model=CatBoostRegressor(random_state=42, verbose=0)),
                )
            )
        if XGBRegressor is not None and _model_enabled(models, "xgboost"):
            specs.append(
                ModelSpec(
                    "xgboost",
                    "regression",
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
        if LGBMRegressor is not None and _model_enabled(models, "lightgbm"):
            specs.append(
                ModelSpec(
                    "lightgbm",
                    "regression",
                    "baseline",
                    _baseline_pipeline(
                        scale_numeric=False,
                        model=LGBMRegressor(random_state=42, n_estimators=300, learning_rate=0.05, verbose=-1),
                    ),
                )
            )
        if _model_enabled(models, "autogluon"):
            estimator = _autogluon_estimator(
                task_type=task_type,
                run_dir=run_dir,
                model_config=(models or {}).get("autogluon"),
            )
            if estimator is not None:
                specs.append(ModelSpec("autogluon", task_type, "baseline", estimator))
        if _model_enabled(models, "tabfm"):
            estimator = _tabfm_estimator(task_type=task_type, model_config=(models or {}).get("tabfm"))
            if estimator is not None:
                specs.append(ModelSpec("tabfm", "regression", "tabfm", estimator))
        return specs

    raise ValueError(f"Unsupported task_type: {task_type}")
