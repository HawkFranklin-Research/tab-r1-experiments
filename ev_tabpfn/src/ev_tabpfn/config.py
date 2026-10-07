from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ev_tabpfn.models.presets import resolve_model_config


VALID_TASKS = {"binary", "multiclass", "regression"}


@dataclass
class DatasetConfig:
    path: str
    name: str | None = None
    target_column: str | None = None
    task: str | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetConfig":
        task = payload.get("task", payload.get("task_override"))
        if task is not None and task not in VALID_TASKS:
            raise ValueError(f"Invalid task for dataset {payload.get('name') or payload.get('path')}: {task}")
        return cls(
            path=str(Path(payload["path"]).expanduser().resolve()),
            name=payload.get("name"),
            target_column=payload.get("target_column"),
            task=task,
            enabled=bool(payload.get("enabled", True)),
        )

    def dataset_id(self) -> str:
        return self.name or Path(self.path).stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.dataset_id(),
            "path": self.path,
            "target_column": self.target_column,
            "task": self.task,
            "enabled": self.enabled,
        }


@dataclass
class EvaluationConfig:
    dataset: DatasetConfig
    output_root: str = "outputs"
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    run_reports: bool = False
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_preset: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        dataset_payload = payload["dataset"] if "dataset" in payload else payload
        model_preset = payload.get("model_preset", payload.get("preset"))
        return cls(
            dataset=DatasetConfig.from_dict(dataset_payload),
            output_root=str(Path(payload.get("output_root", "outputs")).expanduser().resolve()),
            seed=int(payload.get("seed", 42)),
            val_size=float(payload.get("val_size", 0.15)),
            test_size=float(payload.get("test_size", 0.15)),
            run_reports=bool(payload.get("run_reports", False)),
            models=resolve_model_config(payload.get("models"), model_preset),
            model_preset=model_preset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "output_root": self.output_root,
            "seed": self.seed,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "run_reports": self.run_reports,
            "model_preset": self.model_preset,
            "models": self.models,
        }


@dataclass
class BatchEvaluationConfig:
    datasets: list[DatasetConfig]
    run_name: str = "ev_tabpfn_batch"
    output_root: str = "outputs"
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    run_reports: bool = False
    aggregate_after_run: bool = False
    fail_fast: bool = False
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_preset: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchEvaluationConfig":
        datasets = [DatasetConfig.from_dict(item) for item in payload.get("datasets", [])]
        if not datasets:
            raise ValueError("Batch config must contain at least one dataset.")
        model_preset = payload.get("model_preset", payload.get("preset"))
        return cls(
            datasets=datasets,
            run_name=payload.get("run_name", payload.get("batch_name", "ev_tabpfn_batch")),
            output_root=str(Path(payload.get("output_root", "outputs")).expanduser().resolve()),
            seed=int(payload.get("seed", 42)),
            val_size=float(payload.get("val_size", 0.15)),
            test_size=float(payload.get("test_size", 0.15)),
            run_reports=bool(payload.get("run_reports", payload.get("run_phase3", False))),
            aggregate_after_run=bool(payload.get("aggregate_after_run", False)),
            fail_fast=bool(payload.get("fail_fast", False)),
            models=resolve_model_config(payload.get("models"), model_preset),
            model_preset=model_preset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "output_root": self.output_root,
            "seed": self.seed,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "run_reports": self.run_reports,
            "aggregate_after_run": self.aggregate_after_run,
            "fail_fast": self.fail_fast,
            "model_preset": self.model_preset,
            "models": self.models,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }


@dataclass
class EvaluationResult:
    dataset_id: str
    status: str
    run_dir: str | None
    metrics_path: str | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "run_dir": self.run_dir,
            "metrics_path": self.metrics_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class BatchEvaluationResult:
    output_root: str
    manifest_path: str
    summary_path: str
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "counts": self.counts,
        }


def load_json_config(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).expanduser().read_text())
