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
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from analyze_saved_cancer_results import binary_metrics
from resource_limits import configure_process_limits, thread_limit


ROOT = REPO
PACKAGE_SRC = REPO / "ev_tabpfn/src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

AVAILABLE_MODELS = (
    "logistic_regression",
    "random_forest",
    "catboost",
    "xgboost",
    "lightgbm",
    "autogluon",
    "tabpfn_v2",
    "tabpfn_v2_5",
    "tabpfn_v2_6",
    "tabpfn_v3",
    "tabfm_default",
)
FOUNDATION_MODELS = {name for name in AVAILABLE_MODELS if name.startswith(("tabpfn_", "tabfm_"))}


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def preprocessor(*, scale: bool) -> ColumnTransformer:
    from sklearn.compose import make_column_selector

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_exclude=["object", "category"])),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ],
        remainder="drop",
    )


def baseline_estimator(name: str, *, seed: int, threads: int) -> Any:
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)
        return Pipeline([("preprocess", preprocessor(scale=True)), ("model", model)])
    if name == "random_forest":
        model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=threads)
        return Pipeline([("preprocess", preprocessor(scale=False)), ("model", model)])
    if name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(random_seed=seed, thread_count=threads, verbose=0)
        return Pipeline([("preprocess", preprocessor(scale=False)), ("model", model)])
    if name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            random_state=seed,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=threads,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        return Pipeline([("preprocess", preprocessor(scale=False)), ("model", model)])
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.05,
            objective="binary",
            n_jobs=threads,
            verbose=-1,
        )
        return Pipeline([("preprocess", preprocessor(scale=False)), ("model", model)])
    raise ValueError(f"Not a classical baseline: {name}")


def build_estimator(
    name: str,
    *,
    seed: int,
    threads: int,
    run_dir: Path,
    tabfm_backend: str,
    autogluon_time_limit: int,
    device: str = "auto",
) -> Any:
    if name in {"logistic_regression", "random_forest", "catboost", "xgboost", "lightgbm"}:
        return baseline_estimator(name, seed=seed, threads=threads)
    if name == "autogluon":
        from ev_tabpfn.models.autogluon_backend import AutoGluonAdapter

        return AutoGluonAdapter(
            problem_type="binary",
            model_path=run_dir / "autogluon_model",
            presets="medium_quality",
            time_limit=autogluon_time_limit,
            verbosity=0,
        )
    if name == "tabfm_default":
        from ev_tabpfn.models.tabfm_backend import TabFMAdapter

        return TabFMAdapter(
            task_type="binary",
            backend=tabfm_backend,
            ensemble=False,
            max_train_rows=None,
            random_state=seed,
        )
    if name.startswith("tabpfn_"):
        from ev_tabpfn.models.tabpfn_versions import build_tabpfn_classifier

        version = name.removeprefix("tabpfn_")
        config: dict[str, Any] = {"random_state": seed}
        if device != "auto":
            config["device"] = device
        return build_tabpfn_classifier(version, config=config)
    raise ValueError(f"Unknown model: {name}")


def probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    values = np.asarray(estimator.predict_proba(X), dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 2:
        return values[:, 1]
    raise ValueError(f"Expected binary probabilities, got {values.shape}")


def select_threshold(y_validation: np.ndarray, validation_probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([[0.0], validation_probability, [1.0]]))
    scores = np.asarray(
        [balanced_accuracy_score(y_validation, validation_probability >= threshold) for threshold in candidates]
    )
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def resolve_split_path(path_str: str | Path, manifest_dir: Path | None = None) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    # Try finding relative path if manifest_dir is known or standard structure
    parts = p.parts
    for anchor in ("per_cancer", "pooled"):
        if anchor in parts:
            idx = parts.index(anchor)
            rel = Path(*parts[idx:])
            if manifest_dir and (manifest_dir / rel).exists():
                return manifest_dir / rel
            # Search common locations
            for search_base in (FROZEN, Path("generated_folds"), FROZEN):
                if (search_base / rel).exists():
                    return search_base / rel
    return p


def read_split(path: str | Path, manifest_dir: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    resolved = resolve_split_path(path, manifest_dir)
    frame = read_table(resolved)
    if "target" not in frame.columns:
        raise ValueError(f"Target column is missing from {resolved}")
    return frame.drop(columns=["target"]), frame["target"].to_numpy(dtype=int)


def execute_model(
    fold: pd.Series,
    model_name: str,
    *,
    output_root: Path,
    seed: int,
    threads: int,
    tabfm_backend: str,
    autogluon_time_limit: int,
    device: str = "auto",
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    X_train, y_train = read_split(fold["train_path"], manifest_dir)
    X_validation, y_validation = read_split(fold["validation_path"], manifest_dir)
    X_test, y_test = read_split(fold["test_path"], manifest_dir)
    run_dir = (
        output_root
        / str(fold["scope"])
        / str(fold["endpoint"])
        / str(fold["cancer"])
        / f"repeat_{int(fold['repeat']):02d}_fold_{int(fold['fold']):02d}"
        / model_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "scope": fold["scope"],
        "endpoint": fold["endpoint"],
        "cancer": fold["cancer"],
        "repeat": int(fold["repeat"]),
        "fold": int(fold["fold"]),
        "model_name": model_name,
        "n_train": len(y_train),
        "n_validation": len(y_validation),
        "n_test": len(y_test),
    }
    try:
        estimator = build_estimator(
            model_name,
            seed=seed,
            threads=threads,
            run_dir=run_dir,
            tabfm_backend=tabfm_backend,
            autogluon_time_limit=autogluon_time_limit,
            device=device,
        )
        fit_start = time.perf_counter()
        with thread_limit(threads):
            estimator.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        validation_probability = probability(estimator, X_validation)
        test_probability = probability(estimator, X_test)
        predict_time = time.perf_counter() - predict_start
        threshold = select_threshold(y_validation, validation_probability)
        metrics = binary_metrics(y_test, test_probability, threshold=threshold)
        test_metadata = read_table(resolve_split_path(fold["test_metadata_path"], manifest_dir))
        prediction_frame = pd.DataFrame(
            {
                "test_position": np.arange(len(y_test)),
                "sample_id": test_metadata.get("sample_id", pd.Series(np.arange(len(y_test)))).astype(str),
                "patient_id": test_metadata.get("patient_id", pd.Series(np.arange(len(y_test)))).astype(str),
                "cancer_type": test_metadata.get("cancer_type", str(fold["cancer"])),
                "y_true": y_test,
                "probability": test_probability,
                "y_pred": (test_probability >= threshold).astype(int),
            }
        )
        prediction_path = run_dir / "test_predictions.csv"
        prediction_frame.to_csv(prediction_path, index=False)
        row = {
            **base,
            "status": "success",
            "threshold": threshold,
            "threshold_source": "validation split balanced accuracy",
            "fit_time_s": fit_time,
            "predict_time_s": predict_time,
            **metrics,
            "prediction_path": str(prediction_path),
        }
    except Exception as exc:
        traceback_path = run_dir / "traceback.txt"
        traceback_path.write_text(traceback.format_exc())
        row = {
            **base,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_path": str(traceback_path),
        }
    (run_dir / "result.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate models directly on leakage-safe exported folds.")
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", default=",".join(AVAILABLE_MODELS))
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--start-fold", type=int, default=0, help="Start index in fold manifest (0-based)")
    parser.add_argument("--end-fold", type=int, default=None, help="End index in fold manifest (exclusive)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--tabfm-backend", default="pytorch", choices=["jax", "pytorch"])
    parser.add_argument("--autogluon-time-limit", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="PyTorch/CUDA acceleration device (default: auto)")
    parser.add_argument("--resume", action="store_true", help="Skip models on folds that already succeeded")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    args = parser.parse_args()
    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)

    models = parse_list(args.models)
    unknown = sorted(set(models) - set(AVAILABLE_MODELS))
    if unknown:
        parser.error(f"Unknown models: {', '.join(unknown)}")

    manifest_path = Path(args.fold_manifest).expanduser().resolve()
    manifest = pd.read_csv(manifest_path)
    if args.smoke:
        if not args.models or args.models == ",".join(AVAILABLE_MODELS):
            models = ["logistic_regression", "random_forest"]
        args.max_folds = args.max_folds or 1
    elif not args.confirm_full_run and (args.max_folds is None or args.max_folds > 5):
        if set(models) & FOUNDATION_MODELS or len(models) == len(AVAILABLE_MODELS):
            parser.error("Full non-smoke execution across multiple folds requires --confirm-full-run")

    start_idx = args.start_fold
    end_idx = args.end_fold if args.end_fold is not None else len(manifest)
    manifest = manifest.iloc[start_idx:end_idx]
    if args.max_folds is not None:
        manifest = manifest.head(args.max_folds)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_runs = len(manifest) * len(models)
    print(f"Executing {len(manifest)} fold(s) across {len(models)} model(s) [{total_runs} total runs]")

    rows: list[dict[str, Any]] = []
    run_count = 0
    for fold_idx, (_, fold) in enumerate(manifest.iterrows(), start=1):
        fold_desc = f"{fold['scope']}/{fold['endpoint']}/{fold['cancer']}/rep{int(fold['repeat'])}_fold{int(fold['fold'])}"
        for model_name in models:
            run_count += 1
            run_dir = (
                output_root
                / str(fold["scope"])
                / str(fold["endpoint"])
                / str(fold["cancer"])
                / f"repeat_{int(fold['repeat']):02d}_fold_{int(fold['fold']):02d}"
                / model_name
            )
            result_file = run_dir / "result.json"
            if args.resume and result_file.exists():
                try:
                    cached = json.loads(result_file.read_text())
                    if cached.get("status") == "success":
                        print(f"[{run_count}/{total_runs}] SKIP (already completed): {fold_desc} -> {model_name}")
                        rows.append(cached)
                        continue
                except Exception:
                    pass

            print(f"[{run_count}/{total_runs}] RUNNING: {fold_desc} -> {model_name}...")
            res = execute_model(
                fold,
                model_name,
                output_root=output_root,
                seed=args.seed,
                threads=args.threads,
                tabfm_backend=args.tabfm_backend,
                autogluon_time_limit=args.autogluon_time_limit,
                device=args.device,
                manifest_dir=manifest_path.parent,
            )
            status_str = res.get("status", "unknown")
            auc_str = f"AUC={res.get('roc_auc', 0.0):.4f}" if status_str == "success" else f"ERR={res.get('error_type', 'fail')}"
            print(f"[{run_count}/{total_runs}] DONE: {fold_desc} -> {model_name} [{status_str}, {auc_str}]")
            rows.append(res)

    results = pd.DataFrame(rows)
    metrics_path = output_root / "all_fold_model_metrics.csv"
    results.to_csv(metrics_path, index=False)
    (output_root / "run_config.json").write_text(
        json.dumps(
            {
                "fold_manifest": str(Path(args.fold_manifest).expanduser().resolve()),
                "models": models,
                "seed": args.seed,
                "threads": args.threads,
                "memory_gb": args.memory_gb,
                "tabfm_backend": args.tabfm_backend,
                "autogluon_time_limit": args.autogluon_time_limit,
                "device": args.device,
                "smoke": args.smoke,
                "confirmed_full_run": args.confirm_full_run,
                "resume": args.resume,
                "start_fold": start_idx,
                "end_fold": end_idx,
                "max_folds": args.max_folds,
            },
            indent=2,
        )
    )
    print("\nSummary of executed runs:")
    print(results[["scope", "endpoint", "cancer", "repeat", "fold", "model_name", "status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
