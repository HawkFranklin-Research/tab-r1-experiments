from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ev_tabpfn.config import BatchEvaluationConfig, DatasetConfig, EvaluationConfig
from ev_tabpfn.data.formats import (
    create_config_template,
    create_csv_template,
    create_sample_config,
    describe_data_formats,
    format_help_text,
    get_data_format,
)
from ev_tabpfn.data.loader import DataLoader
from ev_tabpfn.models.presets import get_model_preset, list_model_presets, resolve_model_config


def compare_tabpfn_generations(
    datasets: list[str | dict[str, Any]],
    *,
    versions: list[str] | None = None,
    output_root: str = "outputs_generation_compare",
    target_column: str | None = None,
    task: str | None = None,
    seed: int = 42,
    train_rows_cap: int | None = 1024,
    legacy_v1_root: str | None = None,
    model_configs: dict[str, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
):
    from ev_tabpfn.evaluation.generations import compare_tabpfn_generations as _compare
    from ev_tabpfn.evaluation.generations import compare_tabpfn_generations_from_config

    if config is not None:
        return compare_tabpfn_generations_from_config(config)
    return _compare(
        datasets=datasets,
        versions=versions,
        output_root=output_root,
        target_column=target_column,
        task=task,
        seed=seed,
        train_rows_cap=train_rows_cap,
        legacy_v1_root=legacy_v1_root,
        model_configs=model_configs,
    )


def evaluate_dataset(
    dataset_path: str | None = None,
    *,
    target_column: str | None = None,
    task: str | None = None,
    output_root: str = "outputs",
    seed: int = 42,
    run_reports: bool = False,
    models: dict[str, dict[str, Any]] | None = None,
    model_preset: str | None = None,
    config: EvaluationConfig | dict[str, Any] | None = None,
):
    from ev_tabpfn.evaluation.single import evaluate_dataset as _evaluate_dataset

    if config is not None:
        return _evaluate_dataset(config)
    if dataset_path is None:
        raise ValueError("dataset_path is required when config is not provided.")
    return _evaluate_dataset(
        EvaluationConfig(
            dataset=DatasetConfig(path=str(Path(dataset_path).expanduser().resolve()), target_column=target_column, task=task),
            output_root=str(Path(output_root).expanduser().resolve()),
            seed=seed,
            run_reports=run_reports,
            models=resolve_model_config(models, model_preset),
            model_preset=model_preset,
        )
    )


def evaluate_batch(config_path: str | Path | None = None, *, config: BatchEvaluationConfig | dict[str, Any] | None = None, force: bool = False):
    from ev_tabpfn.orchestration.batch import evaluate_batch as _evaluate_batch

    if config is not None:
        return _evaluate_batch(config, force=force)
    if config_path is None:
        raise ValueError("config_path is required when config is not provided.")
    return _evaluate_batch(config_path, force=force)


def aggregate_results(output_root: str | Path | None = None, *, runs_root: str | Path | None = None, results_dir: str | Path | None = None):
    from ev_tabpfn.reporting.aggregator import Aggregator

    if output_root is not None:
        root = Path(output_root)
        runs_root = runs_root or root / "runs"
        results_dir = results_dir or root / "results"
    if runs_root is None or results_dir is None:
        raise ValueError("Provide either output_root or both runs_root and results_dir.")
    return Aggregator(str(runs_root), str(results_dir)).run_aggregation()


def generate_report(run_dir: str | Path) -> dict[str, str]:
    from ev_tabpfn.reporting.reporter import Reporter
    from ev_tabpfn.reporting.visualizer import Visualizer

    run_path = Path(run_dir)
    Visualizer(str(run_path)).run_all()
    report_path = Reporter(str(run_path)).save_report()
    return {"run_dir": str(run_path), "report_path": str(report_path), "plots_dir": str(run_path / "plots_phase3")}


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    import json

    run_path = Path(run_dir)
    metrics_path = run_path / "metrics" / "metrics_summary.json"
    metadata_path = run_path / "metadata" / "dataset_metadata.json"
    status_path = run_path / "metadata" / "model_status.json"
    summary: dict[str, Any] = {"run_dir": str(run_path), "exists": run_path.exists()}
    if metadata_path.exists():
        summary["metadata"] = json.loads(metadata_path.read_text())
    if metrics_path.exists():
        summary["metrics"] = json.loads(metrics_path.read_text())
    if status_path.exists():
        summary["model_status"] = json.loads(status_path.read_text())
    return summary


def infer_task(dataset_path: str, *, target_column: str | None = None) -> str:
    import pandas as pd

    df = pd.read_csv(dataset_path, sep=None, engine="python")
    target = target_column or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    return DataLoader().infer_task(df[target])


def validate_dataset(dataset_path: str, *, target_column: str | None = None, task: str | None = None) -> dict[str, Any]:
    loader = DataLoader()
    dataset = loader.load_local_csv(dataset_path, target_column=target_column, task_override=task)
    return {
        "status": "valid",
        "task_type": dataset.task_type,
        "target_name": dataset.target_name,
        "feature_names": dataset.feature_names,
        "metadata": dataset.metadata,
    }


def _sample_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "sample" / "manifest.json"


def _sample_data_dir() -> Path:
    return _sample_manifest_path().parent


def list_sample_datasets() -> list[dict[str, Any]]:
    import json

    return json.loads(_sample_manifest_path().read_text())["datasets"]


def get_sample_dataset_path(name: str) -> str:
    for dataset in list_sample_datasets():
        if dataset["name"] == name:
            return str(_sample_data_dir() / dataset["file"])
    available = ", ".join(item["name"] for item in list_sample_datasets())
    raise ValueError(f"Unknown sample dataset: {name}. Available: {available}")


def copy_sample_datasets(output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for dataset in list_sample_datasets():
        source = _sample_data_dir() / dataset["file"]
        target = destination / dataset["file"]
        shutil.copy2(source, target)
        copied[dataset["name"]] = str(target)
    return copied
