from __future__ import annotations

import traceback
from pathlib import Path

from ev_tabpfn.artifacts.paths import create_run_dirs
from ev_tabpfn.artifacts.writers import setup_logger, write_json, write_metrics_csv, write_predictions, write_raw_predictions
from ev_tabpfn.config import DatasetConfig, EvaluationConfig, EvaluationResult
from ev_tabpfn.data.loader import DataLoader
from ev_tabpfn.evaluation.labels import ClassificationLabelContract
from ev_tabpfn.evaluation.runner import run_model
from ev_tabpfn.models.registry import build_models
from ev_tabpfn.reporting.plots import plotting_available, save_classification_plots, save_regression_plots


def evaluate_dataset(config: EvaluationConfig | dict) -> EvaluationResult:
    if isinstance(config, dict):
        config = EvaluationConfig.from_dict(config)
    dataset_cfg = config.dataset
    dataset_name = dataset_cfg.dataset_id()
    dirs = create_run_dirs(config.output_root, dataset_name)
    logger = setup_logger(dirs["logs"] / "run.log", logger_name="ev_tabpfn.single")
    logger.info("Starting single-dataset evaluation")
    logger.info("Dataset path: %s", dataset_cfg.path)

    try:
        loader = DataLoader(seed=config.seed)
        dataset = loader.load_local_csv(
            dataset_cfg.path,
            target_column=dataset_cfg.target_column,
            val_size=config.val_size,
            test_size=config.test_size,
            task_override=dataset_cfg.task,
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
                "dataset": dataset_cfg.to_dict(),
                "seed": config.seed,
                "val_size": config.val_size,
                "test_size": config.test_size,
                "models": config.models,
                "plotting_available": plotting_available(),
            },
        )

        summaries: list[dict[str, object]] = []
        model_status: dict[str, object] = {}

        for spec in build_models(dataset.task_type, dataset.y_train, run_dir=dirs["base"], models=config.models):
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
                    (dirs["logs"] / f"{spec.name}_traceback.txt").write_text(result.traceback_text)
                continue

            try:
                write_predictions(
                    dirs["predictions"] / f"{spec.name}_predictions.csv",
                    y_true=dataset.y_test,
                    y_pred=result.y_pred,
                    y_prob=result.y_prob,
                    label_contract=label_contract,
                    y_prob_classes=result.y_prob_classes,
                )
                write_raw_predictions(
                    dirs["base"],
                    model_name=spec.name,
                    dataset_name=dataset_name,
                    task_type=dataset.task_type,
                    y_true=dataset.y_test,
                    y_pred=result.y_pred,
                    y_prob=result.y_prob,
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

        if config.run_reports:
            from ev_tabpfn.reporting.reporter import Reporter
            from ev_tabpfn.reporting.visualizer import Visualizer

            Visualizer(str(dirs["base"])).run_all()
            Reporter(str(dirs["base"])).save_report()

        successful = [row for row in summaries if row["status"] == "success"]
        status = "success" if successful else "failed"
        logger.info("Evaluation finished with status=%s. Output dir: %s", status, dirs["base"])
        return EvaluationResult(
            dataset_id=dataset_name,
            status=status,
            run_dir=str(dirs["base"]),
            metrics_path=str(dirs["metrics"] / "metrics_summary.csv"),
            error_type=None if successful else "NoSuccessfulModels",
            error_message=None if successful else "No model completed successfully.",
        )
    except Exception as exc:
        (dirs["logs"] / "dataset_traceback.txt").write_text(traceback.format_exc())
        logger.exception("Dataset evaluation failed.")
        return EvaluationResult(
            dataset_id=dataset_name,
            status="failed",
            run_dir=str(dirs["base"]),
            metrics_path=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def evaluate_dataset_from_kwargs(
    *,
    dataset_path: str,
    target_column: str | None = None,
    task: str | None = None,
    output_root: str = "outputs",
    seed: int = 42,
    run_reports: bool = False,
    models: dict | None = None,
) -> EvaluationResult:
    return evaluate_dataset(
        EvaluationConfig(
            dataset=DatasetConfig(path=dataset_path, target_column=target_column, task=task),
            output_root=output_root,
            seed=seed,
            run_reports=run_reports,
            models=models or {},
        )
    )
