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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from resource_limits import configure_process_limits, thread_limit


ROOT = REPO
CANCER_ROOT = REPO / "02_cancer_evaluation/historical_split"
PACKAGE_SRC = REPO / "ev_tabpfn/src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

EXPERIMENTS = {
    "per_cancer": CANCER_ROOT / "exp01_per_cancer_fixed_window",
    "pooled_fixed": CANCER_ROOT / "exp02_combined_fixed_window",
    "extreme": CANCER_ROOT / "exp03_combined_extreme_survival",
}
MODEL_METRIC_LOCATIONS = (
    ("tabpfn", "outputs_tabpfn/aggregate/all_model_metrics.csv"),
    ("tabfm", "outputs_tabfm_pytorch/aggregate/all_model_metrics.csv"),
)


def resolve_moved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "cancer-os-exp" in parts:
        relative = Path(*parts[parts.index("cancer-os-exp") + 1 :])
        candidate = CANCER_ROOT / relative
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve artifact path: {value}")


def positive_probability(frame: pd.DataFrame) -> np.ndarray:
    if "prob_1" in frame.columns:
        return frame["prob_1"].to_numpy(dtype=float)
    probability_columns = sorted(col for col in frame.columns if col.startswith("prob_"))
    if len(probability_columns) == 2:
        return frame[probability_columns[1]].to_numpy(dtype=float)
    raise ValueError(f"Binary positive-class probability not found in {list(frame.columns)}")


def encoded_column(frame: pd.DataFrame, preferred: str, fallback: str) -> np.ndarray:
    column = preferred if preferred in frame.columns else fallback
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=int)
    return values.astype(str).map({"0": 0, "1": 1}).to_numpy(dtype=int)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.clip(np.asarray(probability, dtype=float).reshape(-1), 1e-7, 1 - 1e-7)
    y_pred = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def load_existing_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment, root in EXPERIMENTS.items():
        for family, relative in MODEL_METRIC_LOCATIONS:
            path = root / relative
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "experiment", experiment)
            frame["model_family"] = family
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No saved cancer metrics found below {CANCER_ROOT}")
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.loc[result["status"].eq("success")].reset_index(drop=True)


def bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    n = len(y_true)
    attempts = 0
    while len(rows) < iterations and attempts < iterations * 10:
        attempts += 1
        indices = rng.integers(0, n, size=n)
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size != 2:
            continue
        rows.append(binary_metrics(sampled_y, probability[indices]))
    if len(rows) != iterations:
        raise RuntimeError(f"Only {len(rows)} valid bootstrap samples generated from {iterations} requested")
    return rows


def summarize_bootstrap(point: dict[str, float], samples: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric, estimate in point.items():
        values = np.asarray([row[metric] for row in samples], dtype=float)
        result[metric] = estimate
        result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    return result


def analyze_saved_predictions(
    metrics: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for position, row in metrics.iterrows():
        prediction_path = resolve_moved_path(str(row["prediction_path"]))
        predictions = pd.read_csv(prediction_path)
        y_true = encoded_column(predictions, "y_true_encoded", "y_true")
        probability = positive_probability(predictions)
        point = binary_metrics(y_true, probability)
        samples = bootstrap_intervals(
            y_true,
            probability,
            iterations=iterations,
            seed=seed + position,
        )
        interval_rows.append(
            {
                "experiment": row["experiment"],
                "dataset": row["dataset"],
                "endpoint": row.get("endpoint"),
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "n_test": len(y_true),
                "class_0_test": int((y_true == 0).sum()),
                "class_1_test": int((y_true == 1).sum()),
                "bootstrap_iterations": iterations,
                **summarize_bootstrap(point, samples),
                "prediction_path": str(prediction_path),
            }
        )
        prediction_rows.extend(
            {
                "experiment": row["experiment"],
                "dataset": row["dataset"],
                "endpoint": row.get("endpoint"),
                "model_name": row["model_name"],
                "test_position": sample_position,
                "y_true": int(true_value),
                "probability": float(score),
                "prediction_path": str(prediction_path),
            }
            for sample_position, (true_value, score) in enumerate(zip(y_true, probability))
        )
    return pd.DataFrame(interval_rows), pd.DataFrame(prediction_rows)


def make_logistic(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)


def fit_probability(
    estimator: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    *,
    threads: int,
) -> np.ndarray:
    with thread_limit(threads):
        estimator.fit(X_train, y_train)
        return np.asarray(estimator.predict_proba(X_test), dtype=float)[:, 1]


def split_record(record: dict[str, Any]) -> tuple[Any, pd.DataFrame]:
    from ev_tabpfn.data.loader import DataLoader

    dataset_path = resolve_moved_path(record["path"])
    metadata_path = resolve_moved_path(record["metadata_path"])
    dataset = DataLoader(seed=42).load_local_csv(
        str(dataset_path),
        target_column="target",
        task_override="binary",
    )
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != dataset.metadata["total_samples"]:
        raise ValueError(f"Metadata and dataset rows differ for {record['name']}")
    return dataset, metadata


def run_control_models(
    records: list[dict[str, Any]],
    *,
    seed: int,
    threads: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        dataset, metadata = split_record(record)
        train_indices = dataset.y_train.index.to_numpy()
        test_indices = dataset.y_test.index.to_numpy()
        y_train = dataset.y_train.to_numpy(dtype=int)
        y_test = dataset.y_test.to_numpy(dtype=int)

        categorical_models: dict[str, list[str]] = {
            "cancer_identity_only": ["cancer_type"],
        }
        source_column = "cohort_x" if "cohort_x" in metadata.columns else "cohort_y"
        categorical_models["source_identity_only"] = [source_column]
        categorical_models["cancer_and_source_only"] = ["cancer_type", source_column]

        prevalence = float(np.mean(y_train))
        controls: list[tuple[str, np.ndarray]] = [
            ("prevalence_only", np.full(len(y_test), prevalence, dtype=float))
        ]
        for name, columns in categorical_models.items():
            transformer = ColumnTransformer(
                [("categorical", OneHotEncoder(handle_unknown="ignore"), columns)],
                remainder="drop",
            )
            estimator = Pipeline([("features", transformer), ("model", make_logistic(seed))])
            probability = fit_probability(
                estimator,
                metadata.iloc[train_indices],
                y_train,
                metadata.iloc[test_indices],
                threads=threads,
            )
            controls.append((name, probability))

        X_train = dataset.X_train.apply(pd.to_numeric, errors="coerce")
        X_test = dataset.X_test.apply(pd.to_numeric, errors="coerce")
        zero_train = X_train.fillna(0).eq(0).astype(np.int8)
        zero_test = X_test.fillna(0).eq(0).astype(np.int8)
        controls.append(
            (
                "structural_zero_pattern",
                fit_probability(make_logistic(seed), zero_train, y_train, zero_test, threads=threads),
            )
        )
        molecular = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", make_logistic(seed)),
            ]
        )
        controls.append(
            (
                "linear_molecular",
                fit_probability(molecular, X_train, y_train, X_test, threads=threads),
            )
        )

        for control_name, probability in controls:
            rows.append(
                {
                    "experiment": record["experiment"],
                    "dataset": record["name"],
                    "endpoint": record.get("endpoint"),
                    "control": control_name,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "class_0_test": int((y_test == 0).sum()),
                    "class_1_test": int((y_test == 1).sum()),
                    **binary_metrics(y_test, probability),
                }
            )
    return pd.DataFrame(rows)


def load_pooled_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for experiment in ("pooled_fixed", "extreme"):
        manifest = json.loads((EXPERIMENTS[experiment] / "datasets" / "manifest.json").read_text())
        for record in manifest["records"]:
            records.append({"experiment": experiment, **record})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze saved cancer predictions and run lightweight shortcut controls."
    )
    parser.add_argument("--output-dir", default=str(ROOT / "paper" / "tables" / "source_data"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_existing_metrics()
    pooled_records = load_pooled_records()
    if args.smoke:
        metrics = metrics.loc[
            metrics["experiment"].eq("extreme")
            & metrics["model_name"].isin(["tabpfn_v2_5", "tabfm_default"])
        ].head(2)
        pooled_records = [record for record in pooled_records if record["experiment"] == "extreme"]
        args.bootstrap_iterations = min(args.bootstrap_iterations, 50)

    intervals, prediction_index = analyze_saved_predictions(
        metrics,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    metrics.to_csv(output_dir / "cancer_existing_model_metrics.csv", index=False)
    intervals.to_csv(output_dir / "cancer_saved_prediction_bootstrap_intervals.csv", index=False)
    prediction_index.to_csv(output_dir / "cancer_saved_prediction_index.csv", index=False)

    if not args.skip_controls:
        controls = run_control_models(pooled_records, seed=args.seed, threads=args.threads)
        controls.to_csv(output_dir / "cancer_shortcut_control_metrics.csv", index=False)

    (output_dir / "cancer_analysis_config.json").write_text(
        json.dumps(
            {
                "bootstrap_iterations": args.bootstrap_iterations,
                "seed": args.seed,
                "threads": args.threads,
                "memory_gb": args.memory_gb,
                "smoke": args.smoke,
                "foundation_models_executed": False,
                "controls_executed": not args.skip_controls,
            },
            indent=2,
        )
    )
    print(intervals[["experiment", "dataset", "model_name", "n_test", "roc_auc", "pr_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
