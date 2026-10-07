from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, label_binarize


ROOT = REPO
EVALUATOR_ROOT = REPO / "01_benchmark_reproduction/evaluator"
if str(EVALUATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_ROOT))

from scripts.phase1.data_loader import DataLoader  # noqa: E402
from resource_limits import configure_process_limits, thread_limit  # noqa: E402


DATASET_DIR = REPO / "01_benchmark_reproduction/datasets"
DATASETS = [
    "ada_dataset.csv",
    "australian_dataset.csv",
    "blood_transfusion-service-center.csv",
    "car.csv",
    "chum.csv",
    "cmc.csv",
    "credit-g.csv",
]
AVAILABLE_MODELS = (
    "logistic_regression",
    "random_forest",
    "catboost",
    "xgboost",
    "lightgbm",
    "autogluon",
)


def cap_training_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) <= max_rows:
        return X_train, y_train
    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_rows,
        random_state=seed,
        stratify=y_train,
    )
    return X_subset, y_subset


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
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
        [
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_exclude=["object", "category", "bool"])),
            ("cat", Pipeline(categorical_steps), make_column_selector(dtype_include=["object", "category", "bool"])),
        ],
        remainder="drop",
    )


def model_builders(task_type: str, n_classes: int, *, seed: int, threads: int) -> dict[str, Any]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    xgb_kwargs: dict[str, Any] = {
        "random_state": seed,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_jobs": threads,
        "objective": "binary:logistic" if task_type == "binary" else "multi:softprob",
        "eval_metric": "logloss" if task_type == "binary" else "mlogloss",
    }
    if task_type == "multiclass":
        xgb_kwargs["num_class"] = n_classes

    return {
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=threads)),
            ]
        ),
        "catboost": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    CatBoostClassifier(
                        random_seed=seed,
                        verbose=0,
                        thread_count=threads,
                        loss_function="Logloss" if task_type == "binary" else "MultiClass",
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                ("model", XGBClassifier(**xgb_kwargs)),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("prep", build_preprocessor(scale_numeric=False)),
                (
                    "model",
                    LGBMClassifier(
                        random_state=seed,
                        n_estimators=300,
                        learning_rate=0.05,
                        objective="binary" if task_type == "binary" else "multiclass",
                        n_jobs=threads,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def score_predictions(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        if n_classes != 2:
            raise ValueError(f"One-dimensional probabilities are invalid for {n_classes} classes")
        y_prob = np.column_stack([1.0 - y_prob, y_prob])
    if y_prob.ndim != 2 or y_prob.shape != (len(y_true), n_classes):
        raise ValueError(
            f"Probability shape {y_prob.shape} does not match ({len(y_true)}, {n_classes})"
        )
    average = "binary" if task_type == "binary" else "macro"
    if task_type == "binary":
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        pr_auc = average_precision_score(y_true, y_prob[:, 1])
        brier = np.mean((y_prob[:, 1] - y_true) ** 2)
    else:
        labels = np.arange(n_classes)
        y_binary = label_binarize(y_true, classes=labels)
        roc_auc = roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro")
        pr_auc = average_precision_score(y_binary, y_prob, average="macro")
        brier = np.mean(np.sum((y_prob - y_binary) ** 2, axis=1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(log_loss(y_true, y_prob, labels=np.arange(n_classes))),
        "brier": float(brier),
    }


def fit_autogluon(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    task_type: str,
    time_limit: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    from autogluon.tabular import TabularPredictor

    temp_dir = Path(tempfile.mkdtemp(prefix="tabr1-paper-autogluon-"))
    try:
        train_data = X_train.reset_index(drop=True).copy()
        train_data["__target__"] = y_train
        predictor = TabularPredictor(
            label="__target__",
            problem_type=task_type,
            path=str(temp_dir / "model"),
            verbosity=0,
            log_to_file=False,
        )
        fit_start = time.perf_counter()
        predictor.fit(train_data=train_data, time_limit=time_limit, presets="medium_quality")
        fit_time = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        y_pred = np.asarray(predictor.predict(X_test.reset_index(drop=True))).astype(int)
        probability_frame = predictor.predict_proba(
            X_test.reset_index(drop=True),
            as_pandas=True,
            as_multiclass=True,
        )
        y_prob = np.asarray(probability_frame)
        predict_time = time.perf_counter() - predict_start
        return y_pred, y_prob, fit_time, predict_time
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fold-matched classical baselines for the TABR1 manuscript.")
    parser.add_argument("--output-dir", default=str(ROOT / "paper" / "tables" / "source_data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--autogluon-time-limit", type=int, default=30)
    parser.add_argument("--skip-autogluon", action="store_true")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset filenames from the seven-dataset validation set.",
    )
    parser.add_argument(
        "--models",
        default=",".join(AVAILABLE_MODELS),
        help="Comma-separated classical comparators. This script never runs TabPFN or TabFM.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run Australian with logistic regression and random forest on at most 128 training rows.",
    )
    args = parser.parse_args()

    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)
    selected_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    if args.skip_autogluon:
        selected_models = [name for name in selected_models if name != "autogluon"]
    if args.smoke:
        selected_datasets = ["australian_dataset.csv"]
        selected_models = ["logistic_regression", "random_forest"]
        args.max_train_rows = min(args.max_train_rows, 128)
    unknown_datasets = sorted(set(selected_datasets) - set(DATASETS))
    unknown_models = sorted(set(selected_models) - set(AVAILABLE_MODELS))
    if unknown_datasets:
        parser.error(f"Unknown datasets: {', '.join(unknown_datasets)}")
    if unknown_models:
        parser.error(f"Unknown models: {', '.join(unknown_models)}")
    if not selected_models:
        parser.error("At least one model must be selected")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for filename in selected_datasets:
        dataset_path = DATASET_DIR / filename
        dataset = DataLoader(seed=args.seed).load_local_csv(str(dataset_path))
        X_train, y_train_raw = cap_training_rows(
            dataset.X_train,
            dataset.y_train,
            max_rows=args.max_train_rows,
            seed=args.seed,
        )
        encoder = LabelEncoder().fit(y_train_raw)
        y_train = encoder.transform(y_train_raw)
        y_test = encoder.transform(dataset.y_test)
        n_classes = len(encoder.classes_)
        builders = model_builders(dataset.task_type, n_classes, seed=args.seed, threads=args.threads)

        for model_name, estimator in builders.items():
            if model_name not in selected_models:
                continue
            row: dict[str, Any] = {
                "dataset": dataset_path.stem.replace("-", "_"),
                "model": model_name,
                "task_type": dataset.task_type,
                "n_train": len(X_train),
                "n_test": len(dataset.X_test),
                "status": "success",
            }
            try:
                fit_start = time.perf_counter()
                with thread_limit(args.threads):
                    estimator.fit(X_train, y_train)
                row["fit_time_s"] = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                y_pred = np.asarray(estimator.predict(dataset.X_test)).reshape(-1).astype(int)
                y_prob = np.asarray(estimator.predict_proba(dataset.X_test), dtype=float)
                row["predict_time_s"] = time.perf_counter() - predict_start
                row.update(score_predictions(dataset.task_type, y_test, y_pred, y_prob, n_classes))
            except Exception as exc:  # Keep the paper audit complete even if one optional backend fails.
                row.update({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
                metrics_rows.append(row)
                continue
            metrics_rows.append(row)
            for position, (true_value, pred_value, probabilities) in enumerate(zip(y_test, y_pred, y_prob)):
                prediction_rows.append(
                    {
                        "dataset": row["dataset"],
                        "model": model_name,
                        "test_position": position,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                        "probabilities": json.dumps([float(value) for value in probabilities]),
                    }
                )

        if "autogluon" in selected_models:
            model_name = "autogluon"
            row = {
                "dataset": dataset_path.stem.replace("-", "_"),
                "model": model_name,
                "task_type": dataset.task_type,
                "n_train": len(X_train),
                "n_test": len(dataset.X_test),
                "status": "success",
            }
            try:
                y_pred, y_prob, fit_time, predict_time = fit_autogluon(
                    X_train,
                    y_train,
                    dataset.X_test,
                    task_type=dataset.task_type,
                    time_limit=args.autogluon_time_limit,
                )
                row.update({"fit_time_s": fit_time, "predict_time_s": predict_time})
                row.update(score_predictions(dataset.task_type, y_test, y_pred, y_prob, n_classes))
            except Exception as exc:
                row.update({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
                metrics_rows.append(row)
                continue
            metrics_rows.append(row)
            for position, (true_value, pred_value, probabilities) in enumerate(zip(y_test, y_pred, y_prob)):
                prediction_rows.append(
                    {
                        "dataset": row["dataset"],
                        "model": model_name,
                        "test_position": position,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                        "probabilities": json.dumps([float(value) for value in probabilities]),
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(output_dir / "benchmark_matched_baseline_metrics.csv", index=False)
    predictions.to_csv(output_dir / "benchmark_matched_baseline_predictions.csv", index=False)
    successful = metrics.loc[metrics["status"] == "success"]
    summary = (
        successful.groupby("model", as_index=False)[
            ["accuracy", "f1", "roc_auc", "pr_auc", "log_loss", "brier", "fit_time_s", "predict_time_s"]
        ]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    summary.to_csv(output_dir / "benchmark_matched_baseline_mean_metrics.csv", index=False)
    (output_dir / "benchmark_matched_baseline_run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "max_train_rows": args.max_train_rows,
                "threads": args.threads,
                "memory_gb": args.memory_gb,
                "autogluon_time_limit": args.autogluon_time_limit if "autogluon" in selected_models else None,
                "datasets": selected_datasets,
                "models": selected_models,
                "smoke": args.smoke,
                "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
