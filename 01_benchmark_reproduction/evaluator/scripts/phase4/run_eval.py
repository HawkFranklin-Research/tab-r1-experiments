from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase4.discovery import detect_new_run_dir, snapshot_run_dirs
from scripts.phase4.manifest import (
    dataset_signature,
    load_or_init_manifest,
    mark_dataset_finished,
    mark_dataset_started,
    save_manifest,
    should_skip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 batch orchestrator for Evaluate-TABPFN")
    parser.add_argument("--config", required=True, help="Path to a JSON batch config file")
    parser.add_argument("--force", action="store_true", help="Rerun datasets even if already marked successful")
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text())
    if "datasets" not in config or not isinstance(config["datasets"], list):
        raise ValueError("Config must contain a 'datasets' list.")
    return config


def resolve_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    batch_name = raw_config.get("batch_name", "phase4_batch")
    output_root = Path(raw_config.get("output_root", ROOT / "outputs" / batch_name)).resolve()
    batch_autogluon_presets = raw_config.get("autogluon_presets", "medium_quality")
    batch_autogluon_time_limit = float(raw_config.get("autogluon_time_limit", 60.0))
    batch_autogluon_verbosity = int(raw_config.get("autogluon_verbosity", 0))
    datasets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw_config["datasets"]):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset entry at index {index} must be an object.")
        path = str(Path(item["path"]).expanduser().resolve()) if "path" in item else None
        if not path:
            raise ValueError(f"Dataset entry at index {index} is missing 'path'.")
        name = item.get("name") or Path(path).stem
        if name in seen_ids:
            raise ValueError(f"Duplicate dataset name in config: {name}")
        seen_ids.add(name)
        task_override = item.get("task_override")
        if task_override not in {None, "binary", "multiclass", "regression"}:
            raise ValueError(f"Invalid task_override for dataset '{name}': {task_override}")
        datasets.append(
            {
                "name": name,
                "path": path,
                "target_column": item.get("target_column"),
                "task_override": task_override,
                "enabled": bool(item.get("enabled", True)),
                "autogluon_presets": item.get("autogluon_presets", batch_autogluon_presets),
                "autogluon_time_limit": float(item.get("autogluon_time_limit", batch_autogluon_time_limit)),
                "autogluon_verbosity": int(item.get("autogluon_verbosity", batch_autogluon_verbosity)),
            }
        )

    return {
        "batch_name": batch_name,
        "output_root": str(output_root),
        "seed": int(raw_config.get("seed", 42)),
        "run_phase3": bool(raw_config.get("run_phase3", False)),
        "fail_fast": bool(raw_config.get("fail_fast", False)),
        "autogluon_presets": batch_autogluon_presets,
        "autogluon_time_limit": batch_autogluon_time_limit,
        "autogluon_verbosity": batch_autogluon_verbosity,
        "datasets": datasets,
    }


def derive_seed(base_seed: int, dataset_name: str) -> int:
    digest = sha256(f"{base_seed}:{dataset_name}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base_seed) % (2**31 - 1)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def phase2_command(dataset_cfg: dict[str, Any], *, output_root: str, seed: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.phase2.run_single",
        "--dataset-path",
        dataset_cfg["path"],
        "--output-root",
        output_root,
        "--seed",
        str(seed),
        "--autogluon-presets",
        str(dataset_cfg["autogluon_presets"]),
        "--autogluon-time-limit",
        str(dataset_cfg["autogluon_time_limit"]),
        "--autogluon-verbosity",
        str(dataset_cfg["autogluon_verbosity"]),
    ]
    if dataset_cfg.get("target_column"):
        command.extend(["--target-column", dataset_cfg["target_column"]])
    if dataset_cfg.get("task_override"):
        command.extend(["--task-override", dataset_cfg["task_override"]])
    return command


def run_phase3_hooks(run_dir: Path, logger: logging.Logger) -> tuple[str, str | None, str | None]:
    try:
        from scripts.phase3.reporter import Reporter
        from scripts.phase3.visualizer import Visualizer

        Visualizer(str(run_dir)).run_all()
        Reporter(str(run_dir)).save_report()
        logger.info("Phase 3 completed for run_dir=%s", run_dir)
        return "success", None, None
    except Exception as exc:
        logger.exception("Phase 3 failed for run_dir=%s", run_dir)
        return "failed", type(exc).__name__, str(exc)


def run_aggregator(output_root: str, logger: logging.Logger) -> None:
    from scripts.phase3.aggregator import Aggregator

    runs_root = Path(output_root) / "runs"
    results_dir = Path(output_root) / "results"
    Aggregator(str(runs_root), str(results_dir)).run_aggregation()
    logger.info("Aggregation completed for output_root=%s", output_root)


def dataset_log_paths(logs_dir: Path, dataset_id: str) -> tuple[Path, Path]:
    return logs_dir / f"{dataset_id}_phase2_stdout.log", logs_dir / f"{dataset_id}_phase2_stderr.log"


def main() -> int:
    args = parse_args()
    config = resolve_config(load_config(args.config))

    output_root = Path(config["output_root"])
    logs_dir = output_root / "logs"
    summary_dir = output_root / "summary"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "phase4_batch.log")
    logger.info("Starting Phase 4 batch orchestrator")
    logger.info("Config path: %s", args.config)

    write_json(output_root / "batch_config.resolved.json", config)
    manifest_path = output_root / "batch_manifest.json"
    manifest = load_or_init_manifest(
        manifest_path,
        batch_name=config["batch_name"],
        output_root=config["output_root"],
    )
    run_counts = {
        "datasets_total": 0,
        "datasets_success": 0,
        "datasets_failed": 0,
        "datasets_skipped": 0,
    }

    for dataset_cfg in config["datasets"]:
        run_counts["datasets_total"] += 1
        dataset_id = dataset_cfg["name"]
        if not dataset_cfg.get("enabled", True):
            manifest["datasets"][dataset_id] = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_cfg["name"],
                "dataset_path": dataset_cfg["path"],
                "autogluon_presets": dataset_cfg["autogluon_presets"],
                "autogluon_time_limit": dataset_cfg["autogluon_time_limit"],
                "autogluon_verbosity": dataset_cfg["autogluon_verbosity"],
                "status": "skipped",
                "phase2_status": "skipped",
                "phase3_status": "skipped",
                "error_type": None,
                "error_message": "Dataset disabled in config.",
                "last_action": "skipped_disabled",
            }
            run_counts["datasets_skipped"] += 1
            save_manifest(manifest_path, manifest)
            logger.info("Skipping disabled dataset: %s", dataset_id)
            continue

        dataset_seed = derive_seed(config["seed"], dataset_id)
        signature = dataset_signature(dataset_cfg, dataset_seed, config["run_phase3"])
        if should_skip(manifest, dataset_id=dataset_id, signature=signature, force=args.force):
            manifest["datasets"][dataset_id]["last_action"] = "skipped_existing_success"
            run_counts["datasets_skipped"] += 1
            save_manifest(manifest_path, manifest)
            logger.info("Skipping completed dataset: %s", dataset_id)
            continue

        start_time = time.perf_counter()
        mark_dataset_started(
            manifest,
            dataset_id=dataset_id,
            dataset_cfg=dataset_cfg,
            dataset_seed=dataset_seed,
            signature=signature,
        )
        manifest["datasets"][dataset_id]["last_action"] = "running"
        save_manifest(manifest_path, manifest)

        before = snapshot_run_dirs(config["output_root"], dataset_cfg["path"])
        stdout_path, stderr_path = dataset_log_paths(logs_dir, dataset_id)

        phase2_status = "failed"
        phase3_status = "skipped"
        error_type = None
        error_message = None
        run_dir: Path | None = None

        if not Path(dataset_cfg["path"]).exists():
            error_type = "FileNotFoundError"
            error_message = f"Dataset path does not exist: {dataset_cfg['path']}"
            stdout_path.write_text("")
            stderr_path.write_text(error_message + "\n")
            logger.error("Dataset path missing for %s", dataset_id)
        else:
            command = phase2_command(dataset_cfg, output_root=config["output_root"], seed=dataset_seed)
            logger.info("Launching Phase 2 for %s", dataset_id)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            run_dir = detect_new_run_dir(config["output_root"], dataset_cfg["path"], before)
            phase2_status = "success" if completed.returncode == 0 else "failed"
            if completed.returncode != 0:
                error_type = "Phase2RunError"
                error_message = f"Phase 2 returned exit code {completed.returncode}"
                logger.error("Phase 2 failed for %s with exit code %s", dataset_id, completed.returncode)
            else:
                logger.info("Phase 2 completed for %s", dataset_id)

        if phase2_status == "success" and config["run_phase3"] and run_dir is not None:
            phase3_status, phase3_error_type, phase3_error_message = run_phase3_hooks(run_dir, logger)
            if phase3_status != "success":
                error_type = phase3_error_type
                error_message = phase3_error_message
        elif phase2_status == "success" and not config["run_phase3"]:
            phase3_status = "skipped"

        duration_s = time.perf_counter() - start_time
        final_status = "success" if phase2_status == "success" and phase3_status in {"success", "skipped"} else "failed"
        mark_dataset_finished(
            manifest,
            dataset_id=dataset_id,
            status=final_status,
            phase2_status=phase2_status,
            phase3_status=phase3_status,
            run_dir=str(run_dir) if run_dir is not None else None,
            duration_s=duration_s,
            error_type=error_type,
            error_message=error_message,
        )
        manifest["datasets"][dataset_id]["last_action"] = f"completed_{final_status}"
        if final_status == "success":
            run_counts["datasets_success"] += 1
        else:
            run_counts["datasets_failed"] += 1
        save_manifest(manifest_path, manifest)

        if final_status == "failed" and config["fail_fast"]:
            logger.error("Fail-fast enabled. Stopping after dataset: %s", dataset_id)
            break

    if config["run_phase3"]:
        try:
            run_aggregator(config["output_root"], logger)
        except Exception as exc:
            logger.exception("Aggregation failed.")
            manifest["aggregation"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            save_manifest(manifest_path, manifest)
        else:
            manifest["aggregation"] = {"status": "success", "error_type": None, "error_message": None}
            save_manifest(manifest_path, manifest)

    write_json(summary_dir / "batch_summary.json", run_counts)
    logger.info("Phase 4 finished with summary: %s", run_counts)
    return 0 if run_counts["datasets_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
