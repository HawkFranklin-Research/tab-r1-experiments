from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchManifest:
    def __init__(self, path: str | Path, *, run_name: str, output_root: str) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "run_name": run_name,
                "output_root": output_root,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "datasets": {},
            }

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))

    def signature(self, dataset_cfg: dict[str, Any], *, seed: int, run_reports: bool, models: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": dataset_cfg["path"],
            "target_column": dataset_cfg.get("target_column"),
            "task": dataset_cfg.get("task"),
            "seed": seed,
            "run_reports": run_reports,
            "models": models,
        }

    def should_skip(self, dataset_id: str, signature: dict[str, Any], *, force: bool = False) -> bool:
        if force:
            return False
        entry = self.data.get("datasets", {}).get(dataset_id)
        return bool(entry and entry.get("status") == "success" and entry.get("signature") == signature)

    def mark_started(self, dataset_id: str, dataset_cfg: dict[str, Any], *, seed: int, signature: dict[str, Any]) -> None:
        self.data["datasets"][dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_cfg.get("name") or dataset_id,
            "dataset_path": dataset_cfg["path"],
            "target_column": dataset_cfg.get("target_column"),
            "task": dataset_cfg.get("task"),
            "seed": seed,
            "signature": signature,
            "status": "running",
            "run_dir": None,
            "started_at": utc_now(),
            "ended_at": None,
            "duration_s": None,
            "error_type": None,
            "error_message": None,
        }

    def mark_finished(
        self,
        dataset_id: str,
        *,
        status: str,
        run_dir: str | None,
        duration_s: float,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        entry = self.data["datasets"][dataset_id]
        entry["status"] = status
        entry["run_dir"] = run_dir
        entry["ended_at"] = utc_now()
        entry["duration_s"] = duration_s
        entry["error_type"] = error_type
        entry["error_message"] = error_message

    def summarize(self) -> dict[str, int]:
        counts = {
            "datasets_total": 0,
            "datasets_success": 0,
            "datasets_failed": 0,
            "datasets_skipped": 0,
        }
        for entry in self.data.get("datasets", {}).values():
            counts["datasets_total"] += 1
            status = entry.get("status")
            if status == "success":
                counts["datasets_success"] += 1
            elif status == "failed":
                counts["datasets_failed"] += 1
            elif status == "skipped":
                counts["datasets_skipped"] += 1
        return counts

