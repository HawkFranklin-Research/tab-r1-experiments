from __future__ import annotations

from pathlib import Path

from scripts.phase2.artifacts import slugify


def dataset_run_root(output_root: str | Path, dataset_path: str) -> Path:
    dataset_dir = slugify(Path(dataset_path).stem)
    return Path(output_root) / "runs" / dataset_dir


def snapshot_run_dirs(output_root: str | Path, dataset_path: str) -> set[str]:
    run_root = dataset_run_root(output_root, dataset_path)
    if not run_root.exists():
        return set()
    return {path.name for path in run_root.iterdir() if path.is_dir()}


def resolve_run_dir(output_root: str | Path, dataset_path: str, run_name: str) -> Path:
    return dataset_run_root(output_root, dataset_path) / run_name


def detect_new_run_dir(
    output_root: str | Path,
    dataset_path: str,
    before: set[str],
) -> Path | None:
    run_root = dataset_run_root(output_root, dataset_path)
    if not run_root.exists():
        return None

    current_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    created = [path for path in current_dirs if path.name not in before]
    if created:
        return max(created, key=lambda path: path.stat().st_mtime)

    if not current_dirs:
        return None
    return max(current_dirs, key=lambda path: path.stat().st_mtime)
