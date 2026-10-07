from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT / "01_benchmark_reproduction/evaluator"
PHASE1_DIR = EVAL_ROOT / "scripts/phase1"
PHASE2_DIR = EVAL_ROOT / "scripts/phase2"
TABPFN3_DIR = ROOT / "tabpfn_3"

for path in (EVAL_ROOT, PHASE1_DIR, PHASE2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.phase1.data_loader import DataLoader  # noqa: E402
from scripts.phase2.artifacts import create_run_dirs, setup_logger, write_json, write_metrics_csv, write_predictions  # noqa: E402
from scripts.phase2.labels import ClassificationLabelContract  # noqa: E402
from scripts.phase2.metrics import classification_metrics, regression_metrics  # noqa: E402
from scripts.phase2.plots import plotting_available, save_classification_plots, save_regression_plots  # noqa: E402


CLASSIFICATION_DATASETS = [
    "ada_dataset.csv",
    "australian_dataset.csv",
    "blood_transfusion-service-center.csv",
    "car.csv",
    "chum.csv",
    "cmc.csv",
    "credit-g.csv",
]

REGRESSION_DEMO = "linear_relation_2d.csv"

DATASET_ROOT = ROOT / "01_benchmark_reproduction"
CLASSIFICATION_DIR = DATASET_ROOT / "datasets"
REGRESSION_DIR = DATASET_ROOT / "Jupyter Notebook" / "Dataset Type Comparision" / "dummy_datasets"

BINARY_CKPT = TABPFN3_DIR / "tabpfn-v3-classifier-v3_20260417_binary.ckpt"
MULTICLASS_CKPT = TABPFN3_DIR / "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"
REGRESSION_CKPT = TABPFN3_DIR / "tabpfn-v3-regressor-v3_default.ckpt"


@dataclass
class RunSummary:
    dataset_name: str
    task_type: str
    status: str
    output_dir: str
    metrics: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local TabPFN-3 smoke test harness")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "pfn3-test" / "outputs"),
        help="Root directory where run artifacts are written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed",
    )
    parser.add_argument(
        "--include-regression-demo",
        action="store_true",
        help="Also run the small regression demo dataset",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional explicit dataset filenames to run instead of the default set",
    )
    parser.add_argument(
        "--binary-checkpoint",
        default=str(BINARY_CKPT),
        help="Local TabPFN-3 binary checkpoint path",
    )
    parser.add_argument(
        "--multiclass-checkpoint",
        default=str(MULTICLASS_CKPT),
        help="Local TabPFN-3 multiclass checkpoint path",
    )
    parser.add_argument(
        "--regression-checkpoint",
        default=str(REGRESSION_CKPT),
        help="Local TabPFN-3 regression checkpoint path",
    )
    parser.add_argument(
        "--compare-old-tabpfn",
        action="store_true",
        help="Also run TabPFN-2.6 on the same datasets for direct comparison",
    )
    return parser.parse_args()


def _resolve_dataset_path(filename: str) -> Path:
    if filename == REGRESSION_DEMO:
        return REGRESSION_DIR / filename
    return CLASSIFICATION_DIR / filename


def _resolve_checkpoint(task_type: str, *, binary_checkpoint: Path, multiclass_checkpoint: Path, regression_checkpoint: Path) -> Path:
    if task_type == "binary":
        return binary_checkpoint
    if task_type == "multiclass":
        return multiclass_checkpoint
    if task_type == "regression":
        return regression_checkpoint
    raise ValueError(f"Unsupported task type: {task_type}")


def _load_model(task_type: str, *, version: str, checkpoint: Path | None = None):
    if task_type in {"binary", "multiclass"}:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if version == "v2_6":
            return TabPFNClassifier.create_default_for_version(ModelVersion.V2_6)
        if checkpoint is None:
            raise ValueError("Checkpoint is required for TabPFN-3 classification.")
        return TabPFNClassifier(model_path=str(checkpoint))

    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    if version == "v2_6":
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2_6)
    if checkpoint is None:
        raise ValueError("Checkpoint is required for TabPFN-3 regression.")
    return TabPFNRegressor(model_path=str(checkpoint))


def _check_runtime_support() -> None:
    from tabpfn import model_loading

    architectures = getattr(model_loading, "ARCHITECTURES", {})
    if "tabpfn_v3" not in architectures:
        raise RuntimeError(
            "Installed tabpfn runtime does not support 'tabpfn_v3'. "
            "This environment only exposes: "
            f"{', '.join(sorted(architectures.keys())) or 'no architectures'}. "
            "Install a TabPFN runtime build that includes the v3 architecture before rerunning."
        )


def _save_raw_outputs(
    *,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    task_type: str,
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
    y_prob_classes: Any | None,
) -> None:
    raw_dir = output_dir / "raw" / model_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": task_type,
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


def _save_comparison_plot(output_dir: Path, task_type: str, rows: list[dict[str, Any]]) -> str | None:
    if plt is None or len(rows) < 2:
        return None

    models = [str(row["model_name"]) for row in rows if row.get("status") == "success"]
    if len(models) < 2:
        return None

    metric_names = ["accuracy", "roc_auc"] if task_type in {"binary", "multiclass"} else ["rmse", "mae", "r2"]
    available_metrics = [name for name in metric_names if any(row.get(name) is not None for row in rows)]
    if not available_metrics:
        return None

    x = np.arange(len(available_metrics))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, model_name in enumerate(models):
        model_row = next(row for row in rows if row["model_name"] == model_name)
        values = [float(model_row.get(metric)) if model_row.get(metric) is not None else np.nan for metric in available_metrics]
        ax.bar(x + idx * width, values, width=width, label=model_name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(available_metrics)
    ax.set_title(f"TabPFN comparison - {task_type}")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"comparison_{task_type}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def _run_single_dataset(
    *,
    dataset_path: Path,
    output_root: Path,
    seed: int,
    compare_old_tabpfn: bool,
    binary_checkpoint: Path,
    multiclass_checkpoint: Path,
    regression_checkpoint: Path,
) -> RunSummary:
    dataset_name = dataset_path.stem
    dirs = create_run_dirs(output_root, dataset_name)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Starting TabPFN-3 smoke run")
    logger.info("Dataset path: %s", dataset_path)

    loader = DataLoader(seed=seed)
    dataset = loader.load_local_csv(str(dataset_path))

    checkpoint = _resolve_checkpoint(
        dataset.task_type,
        binary_checkpoint=binary_checkpoint,
        multiclass_checkpoint=multiclass_checkpoint,
        regression_checkpoint=regression_checkpoint,
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    write_json(
        dirs["metadata"] / "dataset_metadata.json",
        {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "task_type": dataset.task_type,
            "target_name": dataset.target_name,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
            "checkpoint": str(checkpoint),
            "plotting_available": plotting_available(),
        },
    )

    write_json(
        dirs["metadata"] / "run_config.json",
        {
            "seed": seed,
            "binary_checkpoint": str(binary_checkpoint),
            "multiclass_checkpoint": str(multiclass_checkpoint),
            "regression_checkpoint": str(regression_checkpoint),
            "compare_old_tabpfn": compare_old_tabpfn,
        },
    )

    model_specs: list[tuple[str, Any, Path | None]] = []
    if compare_old_tabpfn:
        model_specs.append(("tabpfn_v2_6", _load_model(dataset.task_type, version="v2_6"), None))
    model_specs.append(("tabpfn_v3", _load_model(dataset.task_type, version="v3", checkpoint=checkpoint), checkpoint))

    summaries: list[dict[str, Any]] = []
    model_status: dict[str, Any] = {}
    saved_plots_all: list[str] = []
    label_contract = ClassificationLabelContract.from_labels(dataset.task_type, dataset.y_train) if dataset.task_type in {"binary", "multiclass"} else None

    for model_name, model, model_checkpoint in model_specs:
        try:
            model.fit(dataset.X_train, dataset.y_train)
            y_pred = model.predict(dataset.X_test)
            y_prob = None
            y_prob_classes = None
            if dataset.task_type in {"binary", "multiclass"} and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(dataset.X_test)
                except Exception:
                    y_prob = None
                y_prob_classes = getattr(model, "classes_", None)
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
        except Exception as exc:
            traceback_path = dirs["logs"] / f"{model_name}_traceback.txt"
            traceback_path.write_text(traceback.format_exc())
            model_status[model_name] = {
                "status": "failed",
                "checkpoint": None if model_checkpoint is None else str(model_checkpoint),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            summaries.append(
                {
                    "model_name": model_name,
                    "task_type": dataset.task_type,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue

        pred_path = dirs["predictions"] / f"{model_name}_predictions.csv"
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
            model_name=model_name,
            dataset_name=dataset_name,
            task_type=dataset.task_type,
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            y_prob_classes=y_prob_classes,
        )

        if dataset.task_type in {"binary", "multiclass"}:
            saved_plots = save_classification_plots(
                output_dir=dirs["plots"],
                task_type=dataset.task_type,
                model_name=model_name,
                y_true=dataset.y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                label_contract=label_contract,
                y_prob_classes=y_prob_classes,
            )
        else:
            saved_plots = save_regression_plots(
                output_dir=dirs["plots"],
                model_name=model_name,
                y_true=dataset.y_test,
                y_pred=y_pred,
            )
        saved_plots_all.extend(saved_plots)
        model_status[model_name] = {
            "status": "success",
            "checkpoint": None if model_checkpoint is None else str(model_checkpoint),
            "plot_files": saved_plots,
        }
        summaries.append(
            {
                "model_name": model_name,
                "task_type": dataset.task_type,
                "status": "success",
                **metrics,
            }
        )

    comparison_plot = _save_comparison_plot(dirs["plots"], dataset.task_type, summaries)
    if comparison_plot is not None:
        saved_plots_all.append(comparison_plot)

    write_metrics_csv(
        dirs["metrics"] / "metrics_summary.csv",
        summaries,
    )
    write_json(
        dirs["metrics"] / "metrics_summary.json",
        {
            "dataset_name": dataset_name,
            "task_type": dataset.task_type,
            "checkpoint": str(checkpoint),
            "rows": summaries,
        },
    )
    write_json(
        dirs["metadata"] / "model_status.json",
        model_status,
    )

    logger.info("Finished TabPFN-3 run: %s", dirs["base"])
    return RunSummary(
        dataset_name=dataset_name,
        task_type=dataset.task_type,
        status="success",
        output_dir=str(dirs["base"]),
        metrics={"rows": summaries},
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    binary_checkpoint = Path(args.binary_checkpoint)
    multiclass_checkpoint = Path(args.multiclass_checkpoint)
    regression_checkpoint = Path(args.regression_checkpoint)

    try:
        _check_runtime_support()
    except Exception as exc:
        summary_payload = {
            "runs": [],
            "counts": {"total": 0, "success": 0, "failed": 0, "missing": 0},
            "runtime_check": {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        }
        write_json(output_root / "batch_summary.json", summary_payload)
        print(json.dumps(summary_payload["runtime_check"], indent=2))
        return 2

    dataset_files = list(args.only) if args.only else list(CLASSIFICATION_DATASETS)
    if args.include_regression_demo and REGRESSION_DEMO not in dataset_files:
        dataset_files.append(REGRESSION_DEMO)

    runs: list[RunSummary] = []
    for filename in dataset_files:
        dataset_path = _resolve_dataset_path(filename)
        if not dataset_path.exists():
            runs.append(
                RunSummary(
                    dataset_name=dataset_path.stem,
                    task_type="unknown",
                    status="missing",
                    output_dir="",
                    metrics={},
                    error_type="FileNotFoundError",
                    error_message=str(dataset_path),
                )
            )
            continue

        try:
            summary = _run_single_dataset(
                dataset_path=dataset_path,
                output_root=output_root,
                seed=args.seed,
                compare_old_tabpfn=args.compare_old_tabpfn,
                binary_checkpoint=binary_checkpoint,
                multiclass_checkpoint=multiclass_checkpoint,
                regression_checkpoint=regression_checkpoint,
            )
        except Exception as exc:
            runs.append(
                RunSummary(
                    dataset_name=dataset_path.stem,
                    task_type="unknown",
                    status="failed",
                    output_dir="",
                    metrics={},
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        runs.append(summary)

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
