from __future__ import annotations

import logging
import time
from hashlib import sha256
from pathlib import Path

from ev_tabpfn.artifacts.manifest import BatchManifest
from ev_tabpfn.artifacts.writers import setup_logger, write_json
from ev_tabpfn.config import BatchEvaluationConfig, BatchEvaluationResult, EvaluationConfig, load_json_config
from ev_tabpfn.evaluation.single import evaluate_dataset
from ev_tabpfn.reporting.aggregator import Aggregator


def derive_seed(base_seed: int, dataset_name: str) -> int:
    digest = sha256(f"{base_seed}:{dataset_name}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base_seed) % (2**31 - 1)


def evaluate_batch(config: BatchEvaluationConfig | dict | str | Path, *, force: bool = False) -> BatchEvaluationResult:
    if isinstance(config, (str, Path)):
        config = BatchEvaluationConfig.from_dict(load_json_config(config))
    elif isinstance(config, dict):
        config = BatchEvaluationConfig.from_dict(config)

    output_root = Path(config.output_root)
    logs_dir = output_root / "logs"
    summary_dir = output_root / "summary"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(logs_dir / "batch.log", logger_name="ev_tabpfn.batch")
    logger.info("Starting batch evaluation")

    write_json(output_root / "batch_config.resolved.json", config.to_dict())
    manifest = BatchManifest(output_root / "batch_manifest.json", run_name=config.run_name, output_root=str(output_root))
    counts = {
        "datasets_total": 0,
        "datasets_success": 0,
        "datasets_failed": 0,
        "datasets_skipped": 0,
    }

    for dataset_cfg in config.datasets:
        counts["datasets_total"] += 1
        dataset_id = dataset_cfg.dataset_id()
        if not dataset_cfg.enabled:
            manifest.data["datasets"][dataset_id] = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_id,
                "dataset_path": dataset_cfg.path,
                "status": "skipped",
                "error_type": None,
                "error_message": "Dataset disabled in config.",
            }
            counts["datasets_skipped"] += 1
            manifest.save()
            continue

        dataset_seed = derive_seed(config.seed, dataset_id)
        signature = manifest.signature(dataset_cfg.to_dict(), seed=dataset_seed, run_reports=config.run_reports, models=config.models)
        if manifest.should_skip(dataset_id, signature, force=force):
            manifest.data["datasets"][dataset_id]["last_action"] = "skipped_existing_success"
            counts["datasets_skipped"] += 1
            manifest.save()
            continue

        start = time.perf_counter()
        manifest.mark_started(dataset_id, dataset_cfg.to_dict(), seed=dataset_seed, signature=signature)
        manifest.save()

        if not Path(dataset_cfg.path).exists():
            result_status = "failed"
            run_dir = None
            error_type = "FileNotFoundError"
            error_message = f"Dataset path does not exist: {dataset_cfg.path}"
            logger.error(error_message)
        else:
            result = evaluate_dataset(
                EvaluationConfig(
                    dataset=dataset_cfg,
                    output_root=str(output_root),
                    seed=dataset_seed,
                    val_size=config.val_size,
                    test_size=config.test_size,
                    run_reports=config.run_reports,
                    models=config.models,
                )
            )
            result_status = result.status
            run_dir = result.run_dir
            error_type = result.error_type
            error_message = result.error_message

        duration = time.perf_counter() - start
        manifest.mark_finished(
            dataset_id,
            status=result_status,
            run_dir=run_dir,
            duration_s=duration,
            error_type=error_type,
            error_message=error_message,
        )
        if result_status == "success":
            counts["datasets_success"] += 1
        else:
            counts["datasets_failed"] += 1
        manifest.save()

        if result_status == "failed" and config.fail_fast:
            logger.error("Fail-fast enabled. Stopping after dataset: %s", dataset_id)
            break

    if config.aggregate_after_run or config.run_reports:
        try:
            outputs = Aggregator(str(output_root / "runs"), str(output_root / "results")).run_aggregation()
            manifest.data["aggregation"] = {"status": "success", "outputs": outputs}
        except Exception as exc:
            logging.exception("Aggregation failed.")
            manifest.data["aggregation"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        manifest.save()

    summary_path = summary_dir / "batch_summary.json"
    write_json(summary_path, counts)
    logger.info("Batch evaluation finished with summary: %s", counts)
    return BatchEvaluationResult(
        output_root=str(output_root),
        manifest_path=str(output_root / "batch_manifest.json"),
        summary_path=str(summary_path),
        counts=counts,
    )

