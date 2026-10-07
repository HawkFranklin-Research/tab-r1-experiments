from __future__ import annotations

from datetime import datetime
from pathlib import Path


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def create_run_dirs(output_root: str | Path, dataset_name: str, run_id: str | None = None) -> dict[str, Path]:
    root = Path(output_root)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / "runs" / slugify(dataset_name) / run_id
    dirs = {
        "base": base,
        "predictions": base / "predictions",
        "metrics": base / "metrics",
        "plots": base / "plots",
        "metadata": base / "metadata",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

