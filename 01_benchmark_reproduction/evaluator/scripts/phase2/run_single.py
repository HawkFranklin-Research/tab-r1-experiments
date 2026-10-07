from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .artifacts import (
    create_run_dirs,
    setup_logger,
    write_json,
    write_metrics_csv,
    write_predictions,
)
from .model_registry import build_models
from .plots import plotting_available, save_classification_plots, save_regression_plots
from .runner import run_model
from .labels import ClassificationLabelContract

ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = ROOT / "scripts" / "phase1"
if str(PHASE1_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE1_DIR))

from data_loader import DataLoader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 single-dataset evaluator")
    parser.add_argument("--dataset-path", required=True, help="Path to a local CSV dataset")
    parser.add_argument("--target-column", default=None, help="Optional target column override")
    parser.add_argument(
        "--task-override",
        choices=["binary", "multiclass", "regression"],
        default=None,
        help="Optional task type override",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--autogluon-presets",
        default="medium_quality",
        help="AutoGluon presets string when AutoGluon is installed",
    )
    parser.add_argument(
        "--autogluon-time-limit",
        type=float,
        default=60.0,
        help="AutoGluon per-dataset time limit in seconds",
    )
    parser.add_argument(
        "--autogluon-verbosity",
        type=int,
        default=0,
        help="AutoGluon verbosity level",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs"),
        help="Root output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_name = Path(args.dataset_path).stem
    dirs = create_run_dirs(args.output_root, dataset_name)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Starting Phase 2 single-dataset evaluation")
    logger.info("Dataset path: %s", args.dataset_path)

    loader = DataLoader(seed=args.seed)
    dataset = loader.load_local_csv(
        args.dataset_path,
        target_column=args.target_column,
        val_size=args.val_size,
        test_size=args.test_size,
        task_override=args.task_override,
    )
    label_contract = None
    if dataset.task_type in {"binary", "multiclass"}:
        label_contract = ClassificationLabelContract.from_labels(dataset.task_type, dataset.y_train)

    write_json(
        dirs["metadata"] / "dataset_metadata.json",
        {
            "dataset_name": dataset_name,
            "task_type": dataset.task_type,
            "target_name": dataset.target_name,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
            "classification_label_contract": label_contract.metadata() if label_contract is not None else None,
        },
    )

    write_json(
        dirs["metadata"] / "run_config.json",
        {
            "dataset_path": args.dataset_path,
            "target_column": args.target_column,
            "task_override": args.task_override,
            "seed": args.seed,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "autogluon_presets": args.autogluon_presets,
            "autogluon_time_limit": args.autogluon_time_limit,
            "autogluon_verbosity": args.autogluon_verbosity,
            "plotting_available": plotting_available(),
        },
    )

    summaries: list[dict[str, object]] = []
    model_status: dict[str, object] = {}

    autogluon_config = {
        "presets": args.autogluon_presets,
        "time_limit": args.autogluon_time_limit,
        "verbosity": args.autogluon_verbosity,
    }

    for spec in build_models(
        dataset.task_type,
        dataset.y_train,
        run_dir=dirs["base"],
        autogluon_config=autogluon_config,
    ):
        logger.info("Running model: %s", spec.name)
        result = run_model(spec, dataset, label_contract=label_contract)
        summaries.append(result.to_summary_row())
        model_status[spec.name] = {
            "status": result.status,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "artifact_status": "pending" if result.status == "success" else "skipped",
            "plot_status": "pending" if result.status == "success" else "skipped",
        }

        if result.status != "success":
            logger.error("Model failed: %s | %s", spec.name, result.error_message)
            if result.traceback_text:
                trace_path = dirs["logs"] / f"{spec.name}_traceback.txt"
                trace_path.write_text(result.traceback_text)
            continue

        pred_path = dirs["predictions"] / f"{spec.name}_predictions.csv"
        try:
            write_predictions(
                pred_path,
                y_true=dataset.y_test,
                y_pred=result.y_pred,
                y_prob=result.y_prob,
                label_contract=label_contract,
                y_prob_classes=result.y_prob_classes,
            )
            model_status[spec.name]["artifact_status"] = "success"
        except Exception as exc:
            model_status[spec.name]["artifact_status"] = "failed"
            model_status[spec.name]["artifact_error_type"] = type(exc).__name__
            model_status[spec.name]["artifact_error_message"] = str(exc)
            (dirs["logs"] / f"{spec.name}_artifact_traceback.txt").write_text(traceback.format_exc())
            logger.error("Artifact save failed: %s | %s", spec.name, exc)
            continue

        try:
            if dataset.task_type in {"binary", "multiclass"}:
                saved_plots = save_classification_plots(
                    output_dir=dirs["plots"],
                    task_type=dataset.task_type,
                    model_name=spec.name,
                    y_true=dataset.y_test,
                    y_pred=result.y_pred,
                    y_prob=result.y_prob,
                    label_contract=label_contract,
                    y_prob_classes=result.y_prob_classes,
                )
            else:
                saved_plots = save_regression_plots(
                    output_dir=dirs["plots"],
                    model_name=spec.name,
                    y_true=dataset.y_test,
                    y_pred=result.y_pred,
                )
            model_status[spec.name]["plot_status"] = "success"
            logger.info("Saved plots for %s: %s", spec.name, saved_plots)
        except Exception as exc:
            model_status[spec.name]["plot_status"] = "failed"
            model_status[spec.name]["plot_error_type"] = type(exc).__name__
            model_status[spec.name]["plot_error_message"] = str(exc)
            (dirs["logs"] / f"{spec.name}_plot_traceback.txt").write_text(traceback.format_exc())
            logger.error("Plot save failed: %s | %s", spec.name, exc)

    write_metrics_csv(dirs["metrics"] / "metrics_summary.csv", summaries)
    write_json(dirs["metrics"] / "metrics_summary.json", {"rows": summaries})
    write_json(dirs["metadata"] / "model_status.json", model_status)

    logger.info("Phase 2 run finished. Output dir: %s", dirs["base"])

    successful = [row for row in summaries if row["status"] == "success"]
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
