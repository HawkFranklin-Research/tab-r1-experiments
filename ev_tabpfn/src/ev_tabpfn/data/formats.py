from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ev_tabpfn.models.presets import resolve_model_config


SUPPORTED_TASKS = ("binary", "multiclass", "regression")


FORMAT_SPECS: dict[str, dict[str, Any]] = {
    "binary": {
        "task": "binary",
        "summary": "Single-target binary classification CSV.",
        "target": "One target column with exactly two unique classes. Labels may be numeric or strings.",
        "features": "One or more feature columns. Numeric and categorical columns are accepted.",
        "example_columns": ["age", "income", "region", "label"],
        "example_rows": [
            [35, 52000, "north", "no"],
            [48, 83000, "south", "yes"],
            [29, 41000, "west", "no"],
            [61, 91000, "east", "yes"],
        ],
        "notes": [
            "If target is omitted, the last CSV column is used.",
            "Binary labels do not need to be 0/1; the evaluator preserves original labels and internally encodes metrics safely.",
        ],
    },
    "multiclass": {
        "task": "multiclass",
        "summary": "Single-target multiclass classification CSV.",
        "target": "One target column with three or more discrete classes.",
        "features": "One or more feature columns. Numeric and categorical columns are accepted.",
        "example_columns": ["buying", "maint", "doors", "safety", "class"],
        "example_rows": [
            ["high", "high", "2", "low", "unacc"],
            ["med", "high", "4", "med", "acc"],
            ["low", "low", "4", "high", "good"],
            ["low", "med", "5more", "high", "vgood"],
        ],
        "notes": [
            "Class labels may be strings or integer-like values.",
            "Probability columns in prediction outputs are named after the original classes.",
        ],
    },
    "regression": {
        "task": "regression",
        "summary": "Single-output regression CSV.",
        "target": "One numeric target column with continuous values.",
        "features": "One or more feature columns. Numeric and categorical columns are accepted by baseline preprocessing.",
        "example_columns": ["x1", "x2", "category", "target"],
        "example_rows": [
            [0.1, 2.0, "a", 4.2],
            [1.4, 0.5, "b", 3.1],
            [2.2, 1.3, "a", 5.8],
            [3.0, 4.1, "c", 9.4],
        ],
        "notes": [
            "Only single-output regression is currently supported.",
            "Multi-target regression requires a future package extension.",
        ],
    },
}


def describe_data_formats() -> dict[str, dict[str, Any]]:
    return FORMAT_SPECS


def get_data_format(task: str) -> dict[str, Any]:
    if task not in FORMAT_SPECS:
        available = ", ".join(SUPPORTED_TASKS)
        raise ValueError(f"Unknown task: {task}. Available tasks: {available}")
    return FORMAT_SPECS[task]


def format_help_text(task: str | None = None) -> str:
    specs = [get_data_format(task)] if task else [FORMAT_SPECS[name] for name in SUPPORTED_TASKS]
    sections = []
    for spec in specs:
        lines = [
            f"Task: {spec['task']}",
            f"Summary: {spec['summary']}",
            f"Target: {spec['target']}",
            f"Features: {spec['features']}",
            f"Example columns: {', '.join(spec['example_columns'])}",
            "Notes:",
        ]
        lines.extend(f"- {note}" for note in spec["notes"])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def create_csv_template(task: str, output_path: str | Path) -> str:
    spec = get_data_format(task)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(spec["example_columns"])
        writer.writerows(spec["example_rows"])
    return str(path)


def create_config_template(
    *,
    output_path: str | Path,
    dataset_path: str | Path,
    target_column: str | None = None,
    task: str | None = None,
    output_root: str | Path = "outputs",
    run_name: str = "ev_tabpfn_run",
    model_preset: str = "smoke",
    run_reports: bool = True,
    aggregate_after_run: bool = True,
) -> str:
    if task is not None:
        get_data_format(task)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": run_name,
        "output_root": str(Path(output_root).expanduser()),
        "seed": 42,
        "run_reports": run_reports,
        "aggregate_after_run": aggregate_after_run,
        "fail_fast": False,
        "model_preset": model_preset,
        "models": resolve_model_config(model_preset=model_preset),
        "datasets": [
            {
                "name": Path(dataset_path).stem,
                "path": str(Path(dataset_path).expanduser()),
                "target_column": target_column,
                "task": task,
            }
        ],
    }
    path.write_text(json.dumps(config, indent=2))
    return str(path)


def create_sample_config(
    *,
    output_path: str | Path,
    samples_dir: str | Path,
    output_root: str | Path = "outputs_sample",
    model_preset: str = "smoke",
) -> str:
    path = Path(output_path).expanduser().resolve()
    samples_root = Path(samples_dir).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = [
        {
            "name": "australian_sample",
            "path": str(samples_root / "australian_sample.csv"),
            "target_column": "target",
            "task": "binary",
        },
        {
            "name": "car_sample",
            "path": str(samples_root / "car_sample.csv"),
            "target_column": "target",
            "task": "multiclass",
        },
        {
            "name": "linear_relation_2d_sample",
            "path": str(samples_root / "linear_relation_2d_sample.csv"),
            "target_column": "y",
            "task": "regression",
        },
    ]
    config = {
        "run_name": "ev_tabpfn_sample_smoke",
        "output_root": str(Path(output_root).expanduser()),
        "seed": 42,
        "run_reports": True,
        "aggregate_after_run": True,
        "fail_fast": False,
        "model_preset": model_preset,
        "models": resolve_model_config(model_preset=model_preset),
        "datasets": datasets,
    }
    path.write_text(json.dumps(config, indent=2))
    return str(path)

