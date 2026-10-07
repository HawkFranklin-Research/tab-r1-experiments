from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "01_benchmark_reproduction/evaluator"
PHASE1_DIR = EVAL_ROOT / "scripts/phase1"
PHASE2_DIR = EVAL_ROOT / "scripts/phase2"

for path in (EVAL_ROOT, PHASE1_DIR, PHASE2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.phase1.data_loader import DataLoader  # noqa: E402
from scripts.phase2.artifacts import create_run_dirs, setup_logger, write_json, write_metrics_csv, write_predictions  # noqa: E402
from scripts.phase2.labels import ClassificationLabelContract  # noqa: E402
from scripts.phase2.metrics import classification_metrics  # noqa: E402
from scripts.phase2.plots import plotting_available, save_classification_plots  # noqa: E402


CLASSIFICATION_DATASETS = [
    "ada_dataset.csv",
    "australian_dataset.csv",
    "blood_transfusion-service-center.csv",
    "car.csv",
    "chum.csv",
    "cmc.csv",
    "credit-g.csv",
]

DATASET_ROOT = ROOT / "01_benchmark_reproduction"
CLASSIFICATION_DIR = DATASET_ROOT / "datasets"
MAX_TRAIN_ROWS = 1024


@dataclass
class RunSummary:
    dataset_name: str
    version: str
    status: str
    output_dir: str
    metrics: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None


class SharedPreprocessor:
    def __init__(self) -> None:
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.numeric_medians: pd.Series | None = None
        self.categorical_map: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame) -> "SharedPreprocessor":
        self.numeric_cols = list(X.select_dtypes(exclude=["object", "category", "bool"]).columns)
        self.categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
        if self.numeric_cols:
            self.numeric_medians = X[self.numeric_cols].apply(pd.to_numeric, errors="coerce").median()
        else:
            self.numeric_medians = pd.Series(dtype=float)
        self.categorical_map = {}
        for col in self.categorical_cols:
            values = X[col].astype("string").fillna("__MISSING__")
            self.categorical_map[col] = sorted(values.unique().tolist())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_cols:
            numeric = X[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(self.numeric_medians)
            parts.append(numeric.to_numpy(dtype=np.float32, copy=False))
        if self.categorical_cols:
            encoded_cols: list[np.ndarray] = []
            for col in self.categorical_cols:
                categories = self.categorical_map.get(col, [])
                mapping = {value: idx for idx, value in enumerate(categories)}
                series = X[col].astype("string").fillna("__MISSING__")
                encoded = series.map(mapping).fillna(-1).to_numpy(dtype=np.float32).reshape(-1, 1)
                encoded_cols.append(encoded)
            parts.extend(encoded_cols)
        if not parts:
            return np.empty((len(X), 0), dtype=np.float32)
        return np.hstack(parts)


def _cap_training_rows(
    X_train: pd.DataFrame,
    y_train: Any,
    *,
    task_type: str,
    seed: int,
) -> tuple[pd.DataFrame, Any]:
    if len(X_train) <= MAX_TRAIN_ROWS:
        return X_train, y_train

    stratify = y_train if task_type in {"binary", "multiclass"} else None
    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_TRAIN_ROWS,
        random_state=seed,
        stratify=stratify,
    )
    return X_subset, y_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TabPFN generations on the same dataset splits")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "pfn3-test" / "outputs_generation_compare"),
        help="Root directory where artifacts are written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=CLASSIFICATION_DATASETS,
        help="Dataset filenames to evaluate",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v2", "v2_5", "v2_6", "v3"],
        help="Model generations to evaluate for the currently imported tabpfn runtime",
    )
    parser.add_argument(
        "--legacy-v1-root",
        default=None,
        help="Path to a TabPFN v1 worktree when running version v1",
    )
    parser.add_argument(
        "--include-v1",
        action="store_true",
        help="Evaluate the legacy v1 model in addition to the current runtime versions",
    )
    return parser.parse_args()


def _resolve_dataset_path(filename: str) -> Path:
    return CLASSIFICATION_DIR / filename


def _load_model(version: str, *, legacy_v1_root: str | None) -> Any:
    if version == "v1":
        if legacy_v1_root is not None and legacy_v1_root not in sys.path:
            sys.path.insert(0, legacy_v1_root)
        from tabpfn import TabPFNClassifier  # type: ignore
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNClassifier(device=device, N_ensemble_configurations=32)

    from tabpfn import TabPFNClassifier  # type: ignore
    from tabpfn.constants import ModelVersion  # type: ignore

    mapping = {
        "v2": ModelVersion.V2,
        "v2_5": ModelVersion.V2_5,
        "v2_6": ModelVersion.V2_6,
        "v3": ModelVersion.V3,
    }
    if version not in mapping:
        raise ValueError(f"Unsupported version: {version}")
    return TabPFNClassifier.create_default_for_version(mapping[version])


def _predict(version: str, model: Any, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if version == "v1":
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        return np.asarray(y_pred), np.asarray(y_prob)
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None
    return np.asarray(y_pred), None if y_prob is None else np.asarray(y_prob)


def _save_raw_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    version: str,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    y_prob_classes: Any | None,
) -> None:
    raw_dir = output_dir / "raw" / version
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name,
        "version": version,
        "y_true": np.asarray(y_true).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_prob": None if y_prob is None else np.asarray(y_prob).tolist(),
        "y_prob_classes": None if y_prob_classes is None else np.asarray(y_prob_classes).tolist(),
    }
    (raw_dir / "raw_predictions.json").write_text(json.dumps(payload, indent=2, default=str))
    np.savez_compressed(
        raw_dir / "raw_predictions.npz",
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob) if y_prob is not None else np.array([]),
    )


def _save_comparison_plot(output_dir: Path, rows: list[dict[str, Any]]) -> str | None:
    if plt is None or len(rows) < 2:
        return None

    models = [str(row["version"]) for row in rows if row.get("status") == "success"]
    if len(models) < 2:
        return None

    metric_names = ["accuracy", "f1", "roc_auc", "log_loss"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, version in enumerate(models):
        row = next(item for item in rows if item["version"] == version and item.get("status") == "success")
        values = [float(row[m]) if row.get(m) is not None else np.nan for m in metric_names]
        ax.bar(x + idx * width, values, width=width, label=version)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_title("TabPFN generation comparison")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "comparison_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def run_dataset(*, dataset_path: Path, output_root: Path, seed: int, versions: list[str], legacy_v1_root: str | None) -> RunSummary:
    dataset_name = dataset_path.stem
    dirs = create_run_dirs(output_root, dataset_name)
    logger = setup_logger(dirs["logs"] / "run.log")
    logger.info("Starting generation comparison for %s", dataset_path)

    loader = DataLoader(seed=seed)
    dataset = loader.load_local_csv(str(dataset_path))
    if dataset.task_type not in {"binary", "multiclass"}:
        raise ValueError(f"Dataset {dataset_name} is not a classification task.")

    X_train_raw, y_train = _cap_training_rows(
        dataset.X_train,
        dataset.y_train,
        task_type=dataset.task_type,
        seed=seed,
    )
    preprocessor = SharedPreprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(dataset.X_test)

    label_contract = ClassificationLabelContract.from_labels(dataset.task_type, y_train)

    write_json(
        dirs["metadata"] / "dataset_metadata.json",
        {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "task_type": dataset.task_type,
            "target_name": dataset.target_name,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
            "plotting_available": plotting_available(),
            "versions": versions,
            "train_rows_cap": MAX_TRAIN_ROWS,
            "train_rows_used": int(len(X_train_raw)),
        },
    )

    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {}

    for version in versions:
        model = None
        try:
            model = _load_model(version, legacy_v1_root=legacy_v1_root)
            if version == "v1":
                model.fit(X_train, y_train, overwrite_warning=True)
            else:
                model.fit(X_train, y_train)
            y_pred, y_prob = _predict(version, model, X_test)
            y_prob_classes = getattr(model, "classes_", None)
            metrics = classification_metrics(
                dataset.task_type,
                dataset.y_test,
                y_pred,
                y_prob,
                label_contract=label_contract,
                y_prob_classes=y_prob_classes,
            )
        except Exception as exc:
            status[version] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            (dirs["logs"] / f"{version}_traceback.txt").write_text(traceback.format_exc())
            rows.append(
                {
                    "version": version,
                    "task_type": dataset.task_type,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        pred_path = dirs["predictions"] / f"{version}_predictions.csv"
        write_predictions(
            pred_path,
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            label_contract=label_contract,
            y_prob_classes=y_prob_classes,
        )
        _save_raw_outputs(
            output_dir=dirs["base"],
            dataset_name=dataset_name,
            version=version,
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            y_prob_classes=y_prob_classes,
        )

        saved_plots = save_classification_plots(
            output_dir=dirs["plots"],
            task_type=dataset.task_type,
            model_name=version,
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            label_contract=label_contract,
            y_prob_classes=y_prob_classes,
        )
        status[version] = {"status": "success", "plot_files": saved_plots}
        rows.append(
            {
                "version": version,
                "task_type": dataset.task_type,
                "status": "success",
                **metrics,
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_plot = _save_comparison_plot(dirs["plots"], rows)
    if comparison_plot is not None:
        status["comparison_plot"] = comparison_plot

    write_metrics_csv(dirs["metrics"] / "metrics_summary.csv", rows)
    write_json(dirs["metrics"] / "metrics_summary.json", {"dataset_name": dataset_name, "rows": rows})
    write_json(dirs["metadata"] / "model_status.json", status)
    return RunSummary(dataset_name=dataset_name, version="batch", status="success", output_dir=str(dirs["base"]), metrics={"rows": rows})


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    versions = list(args.versions)
    if args.include_v1 and "v1" not in versions:
        versions = ["v1"] + versions

    runs: list[RunSummary] = []
    for dataset_name in args.datasets:
        dataset_path = _resolve_dataset_path(dataset_name)
        if not dataset_path.exists():
            runs.append(
                RunSummary(
                    dataset_name=dataset_path.stem,
                    version="batch",
                    status="missing",
                    output_dir="",
                    metrics={},
                    error_type="FileNotFoundError",
                    error_message=str(dataset_path),
                )
            )
            continue
        try:
            summary = run_dataset(
                dataset_path=dataset_path,
                output_root=output_root,
                seed=args.seed,
                versions=versions,
                legacy_v1_root=args.legacy_v1_root,
            )
            runs.append(summary)
        except Exception as exc:
            runs.append(
                RunSummary(
                    dataset_name=dataset_path.stem,
                    version="batch",
                    status="failed",
                    output_dir="",
                    metrics={},
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    summary_payload = {
        "runs": [run.__dict__ for run in runs],
        "counts": {
            "total": len(runs),
            "success": sum(1 for run in runs if run.status == "success"),
            "failed": sum(1 for run in runs if run.status == "failed"),
            "missing": sum(1 for run in runs if run.status == "missing"),
        },
    }
    write_json(output_root / "batch_summary.json", summary_payload)
    print(json.dumps(summary_payload["counts"], indent=2))
    return 0 if summary_payload["counts"]["failed"] == 0 and summary_payload["counts"]["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
