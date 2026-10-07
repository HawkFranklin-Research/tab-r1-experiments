from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "tabfm": {"enabled": False},
        "tabpfn": {"enabled": False},
        "autogluon": {"enabled": False},
        "catboost": {"enabled": False},
        "xgboost": {"enabled": False},
        "lightgbm": {"enabled": False},
        "random_forest": {"enabled": True},
        "logistic_regression": {"enabled": True},
        "ridge": {"enabled": True},
    },
    "standard": {
        "tabfm": {"enabled": False},
        "tabpfn": {"enabled": False},
        "autogluon": {"enabled": False},
        "catboost": {"enabled": True},
        "xgboost": {"enabled": True},
        "lightgbm": {"enabled": True},
        "random_forest": {"enabled": True},
        "logistic_regression": {"enabled": True},
        "ridge": {"enabled": True},
    },
    "full": {
        "tabfm": {"enabled": False},
        "tabpfn": {"enabled": True},
        "autogluon": {
            "enabled": True,
            "presets": "medium_quality",
            "time_limit": 60,
            "verbosity": 0,
        },
        "catboost": {"enabled": True},
        "xgboost": {"enabled": True},
        "lightgbm": {"enabled": True},
        "random_forest": {"enabled": True},
        "logistic_regression": {"enabled": True},
        "ridge": {"enabled": True},
    },
    "tabpfn-generation": {
        "tabfm": {"enabled": False},
        "tabpfn": {"enabled": False},
        "tabpfn_v3": {"enabled": True},
        "tabpfn_v2": {"enabled": False},
        "tabpfn_v2_5": {"enabled": False},
        "tabpfn_v2_6": {"enabled": False},
        "autogluon": {"enabled": False},
        "catboost": {"enabled": False},
        "xgboost": {"enabled": False},
        "lightgbm": {"enabled": False},
        "random_forest": {"enabled": False},
        "logistic_regression": {"enabled": False},
        "ridge": {"enabled": False},
    },
    "tabfm": {
        "tabfm": {
            "enabled": True,
            "backend": "jax",
            "ensemble": False,
            "max_train_rows": 1024,
            "n_estimators": 1,
            "batch_size": 1,
            "num_folds_for_cv": 2,
            "load_kwargs": {
                "col_attention_impl": "jax",
                "row_attention_impl": "jax",
                "icl_attention_impl": "jax",
            },
        },
        "tabpfn": {"enabled": False},
        "tabpfn_v3": {"enabled": False},
        "tabpfn_v2": {"enabled": False},
        "tabpfn_v2_5": {"enabled": False},
        "tabpfn_v2_6": {"enabled": False},
        "autogluon": {"enabled": False},
        "catboost": {"enabled": False},
        "xgboost": {"enabled": False},
        "lightgbm": {"enabled": False},
        "random_forest": {"enabled": False},
        "logistic_regression": {"enabled": False},
        "ridge": {"enabled": False},
    },
    "tabfm-ensemble": {
        "tabfm": {
            "enabled": True,
            "backend": "jax",
            "ensemble": True,
            "max_train_rows": 1024,
            "load_kwargs": {
                "col_attention_impl": "jax",
                "row_attention_impl": "jax",
                "icl_attention_impl": "jax",
            },
        },
        "tabpfn": {"enabled": False},
        "tabpfn_v3": {"enabled": False},
        "tabpfn_v2": {"enabled": False},
        "tabpfn_v2_5": {"enabled": False},
        "tabpfn_v2_6": {"enabled": False},
        "autogluon": {"enabled": False},
        "catboost": {"enabled": False},
        "xgboost": {"enabled": False},
        "lightgbm": {"enabled": False},
        "random_forest": {"enabled": False},
        "logistic_regression": {"enabled": False},
        "ridge": {"enabled": False},
    },
    "foundation": {
        "tabfm": {
            "enabled": True,
            "backend": "jax",
            "ensemble": False,
            "max_train_rows": 1024,
            "n_estimators": 1,
            "batch_size": 1,
            "num_folds_for_cv": 2,
            "load_kwargs": {
                "col_attention_impl": "jax",
                "row_attention_impl": "jax",
                "icl_attention_impl": "jax",
            },
        },
        "tabpfn": {"enabled": False},
        "tabpfn_v3": {"enabled": True},
        "tabpfn_v2": {"enabled": False},
        "tabpfn_v2_5": {"enabled": False},
        "tabpfn_v2_6": {"enabled": False},
        "autogluon": {"enabled": False},
        "catboost": {"enabled": False},
        "xgboost": {"enabled": False},
        "lightgbm": {"enabled": False},
        "random_forest": {"enabled": False},
        "logistic_regression": {"enabled": False},
        "ridge": {"enabled": False},
    },
}


def list_model_presets() -> dict[str, dict[str, Any]]:
    return {
        "smoke": {
            "description": "Fast local sanity check using only lightweight sklearn baselines.",
            "models": deepcopy(MODEL_PRESETS["smoke"]),
        },
        "standard": {
            "description": "Benchmark-oriented GBM/sklearn baseline set without TabPFN or AutoGluon.",
            "models": deepcopy(MODEL_PRESETS["standard"]),
        },
        "full": {
            "description": "Full evaluator set including TabPFN and AutoGluon; requires heavier runtime and TabPFN token for TabPFN.",
            "models": deepcopy(MODEL_PRESETS["full"]),
        },
        "tabpfn-generation": {
            "description": "Versioned TabPFN model preset. Defaults to TabPFN v3 for focused TabPFN-only runs.",
            "models": deepcopy(MODEL_PRESETS["tabpfn-generation"]),
        },
        "tabfm": {
            "description": "TabFM-only zero-shot foundation model run. Requires an installed TabFM backend and model-weight access.",
            "models": deepcopy(MODEL_PRESETS["tabfm"]),
        },
        "tabfm-ensemble": {
            "description": "TabFM ensemble run using feature crosses, SVD features, NNLS blending, and calibration.",
            "models": deepcopy(MODEL_PRESETS["tabfm-ensemble"]),
        },
        "foundation": {
            "description": "Foundation-model run with TabFM and TabPFN v3 enabled. Requires both runtimes.",
            "models": deepcopy(MODEL_PRESETS["foundation"]),
        },
    }


def get_model_preset(name: str) -> dict[str, dict[str, Any]]:
    if name not in MODEL_PRESETS:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset: {name}. Available presets: {available}")
    return deepcopy(MODEL_PRESETS[name])


def resolve_model_config(
    models: dict[str, dict[str, Any]] | None = None,
    model_preset: str | None = None,
) -> dict[str, dict[str, Any]]:
    if model_preset is None:
        return dict(models or {})

    resolved = get_model_preset(model_preset)
    for name, config in (models or {}).items():
        existing = resolved.get(name, {})
        merged = {**existing, **config}
        resolved[name] = merged
    return resolved
