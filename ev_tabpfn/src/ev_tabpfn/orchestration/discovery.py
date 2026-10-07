from __future__ import annotations

from pathlib import Path

from ev_tabpfn.artifacts.paths import slugify


def dataset_run_root(output_root: str | Path, dataset_name: str) -> Path:
    return Path(output_root) / "runs" / slugify(dataset_name)


def latest_run_dir(output_root: str | Path, dataset_name: str) -> Path | None:
    root = dataset_run_root(output_root, dataset_name)
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return max(dirs, key=lambda path: path.stat().st_mtime) if dirs else None

