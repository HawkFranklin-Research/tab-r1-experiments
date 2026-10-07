from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from ev_tabpfn.artifacts.paths import create_run_dirs
from ev_tabpfn.artifacts.writers import setup_logger, write_json, write_metrics_csv, write_predictions, write_raw_predictions
from ev_tabpfn.data.loader import DataLoader
from ev_tabpfn.evaluation.generation_preprocessing import TabPFNGenerationPreprocessor, cap_training_rows
from ev_tabpfn.evaluation.labels import ClassificationLabelContract
from ev_tabpfn.evaluation.metrics import classification_metrics
from ev_tabpfn.models.tabpfn_versions import (
    DEFAULT_TABPFN_VERSION,
    build_tabpfn_classifier,
    clear_accelerator_cache,
    normalize_tabpfn_version,
    purge_tabpfn_modules,
)
from ev_tabpfn.reporting.plots import plotting_available, save_classification_plots


@dataclass
class GenerationComparisonResult:
    output_root: str
    summary_path: str
    aggregate_path: str
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "summary_path": self.summary_path,
            "aggregate_path": self.aggregate_path,
            "counts": self.counts,
        }


def _as_dataset_items(datasets: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in datasets:
        if isinstance(item, dict):
            path = item["path"]
            items.append(
                {
                    "path": str(Path(path).expanduser().resolve()),
                    "name": item.get("name") or Path(path).stem,
                    "target_column": item.get("target_column"),
                    "task": item.get("task"),
                }
            )
        else:
            items.append(
                {
                    "path": str(Path(item).expanduser().resolve()),
                    "name": Path(item).stem,
                    "target_column": None,
                    "task": None,
                }
            )
    return items


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


def _save_comparison_plot(output_dir: Path, rows: list[dict[str, Any]]) -> str | None:
    if plt is None:
        return None
    successful = [row for row in rows if row.get("status") == "success"]
    if len(successful) < 2:
        return None
    metric_names = ["accuracy", "f1", "roc_auc", "log_loss"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(successful)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, row in enumerate(successful):
        values = [float(row[m]) if row.get(m) is not None else np.nan for m in metric_names]
        ax.bar(x + idx * width, values, width=width, label=str(row["model_name"]))
    ax.set_xticks(x + width * (len(successful) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_title("TabPFN generation comparison")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "generation_comparison_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _run_one_dataset(
    *,
    dataset_item: dict[str, Any],
    versions: list[str],
    output_root: Path,
    seed: int,
    train_rows_cap: int | None,
    legacy_v1_root: str | None,
    model_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dirs = create_run_dirs(output_root, dataset_item["name"])
    logger = setup_logger(dirs["logs"] / "run.log", logger_name="ev_tabpfn.generations")
    loader = DataLoader(seed=seed)
    dataset = loader.load_local_csv(
        dataset_item["path"],
        target_column=dataset_item.get("target_column"),
        task_override=dataset_item.get("task"),
    )
    if dataset.task_type not in {"binary", "multiclass"}:
        raise ValueError("Generation comparison currently supports binary and multiclass classification.")

    capped = cap_training_rows(dataset.X_train, dataset.y_train, task_type=dataset.task_type, seed=seed, max_rows=train_rows_cap)
    preprocessor = TabPFNGenerationPreprocessor().fit(capped.X_train)
    X_train = preprocessor.transform(capped.X_train)
    X_test = preprocessor.transform(dataset.X_test)
    label_contract = ClassificationLabelContract.from_labels(dataset.task_type, capped.y_train)

    write_json(
        dirs["metadata"] / "dataset_metadata.json",
        {
            "dataset_name": dataset_item["name"],
            "dataset_path": dataset_item["path"],
            "task_type": dataset.task_type,
            "target_name": dataset.target_name,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
            "versions": versions,
            "train_rows_cap": train_rows_cap,
            "train_rows_used": capped.rows_used,
            "train_rows_original": capped.rows_original,
            "classification_label_contract": label_contract.metadata(),
            "plotting_available": plotting_available(),
        },
    )

    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    for version in versions:
        model = None
        model_name = f"tabpfn_{version}"
        config = dict(model_configs.get(model_name, model_configs.get(version, {})))
        if version == "v1" and legacy_v1_root:
            config.setdefault("legacy_v1_root", legacy_v1_root)
            config.setdefault("allow_runtime_swap", True)
        if version != "v1":
            if legacy_v1_root:
                legacy_path = str(Path(legacy_v1_root).expanduser().resolve())
                if legacy_path in sys.path:
                    sys.path.remove(legacy_path)
            purge_tabpfn_modules()
        try:
            model = build_tabpfn_classifier(version, config)
            if version == "v1":
                model.fit(X_train, capped.y_train, overwrite_warning=True)
            else:
                model.fit(X_train, capped.y_train)
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
            status[model_name] = {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}
            (dirs["logs"] / f"{model_name}_traceback.txt").write_text(traceback.format_exc())
            rows.append(
                {
                    "model_name": model_name,
                    "version": version,
                    "task_type": dataset.task_type,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            clear_accelerator_cache()
            continue

        write_predictions(
            dirs["predictions"] / f"{model_name}_predictions.csv",
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            label_contract=label_contract,
            y_prob_classes=y_prob_classes,
        )
        raw_paths = write_raw_predictions(
            dirs["base"],
            model_name=model_name,
            dataset_name=dataset_item["name"],
            task_type=dataset.task_type,
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            y_prob_classes=y_prob_classes,
        )
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
        status[model_name] = {"status": "success", "plot_files": saved_plots, "raw_files": raw_paths}
        rows.append({"model_name": model_name, "version": version, "task_type": dataset.task_type, "status": "success", **metrics})
        logger.info("Completed %s", model_name)
        del model
        clear_accelerator_cache()

    comparison_plot = _save_comparison_plot(dirs["plots"], rows)
    if comparison_plot:
        status["comparison_plot"] = comparison_plot
    write_metrics_csv(dirs["metrics"] / "metrics_summary.csv", rows)
    write_json(dirs["metrics"] / "metrics_summary.json", {"dataset_name": dataset_item["name"], "rows": rows})
    write_json(dirs["metadata"] / "model_status.json", status)
    return {"dataset": dataset_item["name"], "status": "success", "run_dir": str(dirs["base"]), "rows": rows}


def compare_tabpfn_generations(
    datasets: list[str | dict[str, Any]],
    *,
    versions: list[str] | None = None,
    output_root: str | Path = "outputs_generation_compare",
    target_column: str | None = None,
    task: str | None = None,
    seed: int = 42,
    train_rows_cap: int | None = 1024,
    legacy_v1_root: str | None = None,
    model_configs: dict[str, dict[str, Any]] | None = None,
) -> GenerationComparisonResult:
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_versions = [normalize_tabpfn_version(item) for item in (versions or [DEFAULT_TABPFN_VERSION])]
    dataset_items = _as_dataset_items(datasets)
    for item in dataset_items:
        item["target_column"] = item.get("target_column") or target_column
        item["task"] = item.get("task") or task

    runs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for item in dataset_items:
        try:
            run = _run_one_dataset(
                dataset_item=item,
                versions=normalized_versions,
                output_root=output_path,
                seed=seed,
                train_rows_cap=train_rows_cap,
                legacy_v1_root=legacy_v1_root,
                model_configs=model_configs or {},
            )
            runs.append(run)
            for row in run["rows"]:
                metric_rows.append({"dataset": item["name"], **row})
        except Exception as exc:
            runs.append({"dataset": item["name"], "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})

    summary_path = output_path / "generation_summary.json"
    counts = {
        "datasets_total": len(runs),
        "datasets_success": sum(1 for run in runs if run["status"] == "success"),
        "datasets_failed": sum(1 for run in runs if run["status"] == "failed"),
    }
    write_json(summary_path, {"counts": counts, "runs": runs, "versions": normalized_versions})

    aggregate_dir = output_path / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / "generation_mean_metrics.csv"
    metric_df = pd.DataFrame(metric_rows)
    if not metric_df.empty:
        metric_df.to_csv(aggregate_dir / "generation_dataset_metrics.csv", index=False)
        successful = metric_df[metric_df["status"] == "success"]
        mean_df = successful.groupby("model_name", as_index=False)[["accuracy", "f1", "roc_auc", "log_loss"]].mean()
        mean_df.to_csv(aggregate_path, index=False)
    else:
        pd.DataFrame(columns=["model_name", "accuracy", "f1", "roc_auc", "log_loss"]).to_csv(aggregate_path, index=False)

    return GenerationComparisonResult(
        output_root=str(output_path),
        summary_path=str(summary_path),
        aggregate_path=str(aggregate_path),
        counts=counts,
    )


def compare_tabpfn_generations_from_config(config: dict[str, Any]) -> GenerationComparisonResult:
    return compare_tabpfn_generations(
        datasets=config["datasets"],
        versions=config.get("versions"),
        output_root=config.get("output_root", "outputs_generation_compare"),
        target_column=config.get("target_column"),
        task=config.get("task"),
        seed=int(config.get("seed", 42)),
        train_rows_cap=config.get("train_rows_cap", 1024),
        legacy_v1_root=config.get("legacy_v1_root"),
        model_configs=config.get("models", {}),
    )
