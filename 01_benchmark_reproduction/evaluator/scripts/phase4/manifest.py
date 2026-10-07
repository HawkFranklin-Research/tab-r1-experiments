from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_or_init_manifest(path: str | Path, *, batch_name: str, output_root: str) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {
        "batch_name": batch_name,
        "output_root": output_root,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "datasets": {},
    }


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    Path(path).write_text(json.dumps(manifest, indent=2))


def dataset_signature(dataset_cfg: dict[str, Any], seed: int, run_phase3: bool) -> dict[str, Any]:
    return {
        "path": dataset_cfg["path"],
        "target_column": dataset_cfg.get("target_column"),
        "task_override": dataset_cfg.get("task_override"),
        "seed": seed,
        "run_phase3": run_phase3,
        "autogluon_presets": dataset_cfg.get("autogluon_presets"),
        "autogluon_time_limit": dataset_cfg.get("autogluon_time_limit"),
        "autogluon_verbosity": dataset_cfg.get("autogluon_verbosity"),
    }


def get_entry(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any] | None:
    return manifest.get("datasets", {}).get(dataset_id)


def should_skip(
    manifest: dict[str, Any],
    *,
    dataset_id: str,
    signature: dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return False
    entry = get_entry(manifest, dataset_id)
    if not entry:
        return False
    if entry.get("status") != "success":
        return False
    return entry.get("signature") == signature


def mark_dataset_started(
    manifest: dict[str, Any],
    *,
    dataset_id: str,
    dataset_cfg: dict[str, Any],
    dataset_seed: int,
    signature: dict[str, Any],
) -> None:
    manifest["datasets"][dataset_id] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_cfg["name"],
        "dataset_path": dataset_cfg["path"],
        "target_column": dataset_cfg.get("target_column"),
        "task_override": dataset_cfg.get("task_override"),
        "autogluon_presets": dataset_cfg.get("autogluon_presets"),
        "autogluon_time_limit": dataset_cfg.get("autogluon_time_limit"),
        "autogluon_verbosity": dataset_cfg.get("autogluon_verbosity"),
        "seed": dataset_seed,
        "signature": signature,
        "status": "running",
        "phase2_status": "pending",
        "phase3_status": "pending",
        "run_dir": None,
        "started_at": utc_now(),
        "ended_at": None,
        "duration_s": None,
        "error_type": None,
        "error_message": None,
    }


def mark_dataset_finished(
    manifest: dict[str, Any],
    *,
    dataset_id: str,
    status: str,
    phase2_status: str,
    phase3_status: str,
    run_dir: str | None,
    duration_s: float,
    error_type: str | None,
    error_message: str | None,
) -> None:
    entry = manifest["datasets"][dataset_id]
    entry["status"] = status
    entry["phase2_status"] = phase2_status
    entry["phase3_status"] = phase3_status
    entry["run_dir"] = run_dir
    entry["ended_at"] = utc_now()
    entry["duration_s"] = duration_s
    entry["error_type"] = error_type
    entry["error_message"] = error_message


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {
        "datasets_total": 0,
        "datasets_success": 0,
        "datasets_failed": 0,
        "datasets_skipped": 0,
    }
    for entry in manifest.get("datasets", {}).values():
        counts["datasets_total"] += 1
        status = entry.get("status")
        if status == "success":
            counts["datasets_success"] += 1
        elif status == "failed":
            counts["datasets_failed"] += 1
        elif status == "skipped":
            counts["datasets_skipped"] += 1
    return counts
