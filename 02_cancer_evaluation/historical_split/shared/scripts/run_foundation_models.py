from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score

SHARED = Path("/home/prime/Documents/g3/cancer-os-exp/shared/scripts")
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from os_exp_common import DEFAULT_PACKAGE_SRC, class_counts  # noqa: E402


def _ensure_package_import(package_src: Path) -> None:
    package_src = package_src.expanduser().resolve()
    if str(package_src) not in sys.path:
        sys.path.insert(0, str(package_src))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in manifest: {path}")
    return records


def _split_stats(dataset_path: str, metadata_path: str | None, package_src: Path, seed: int) -> dict[str, Any]:
    _ensure_package_import(package_src)
    from ev_tabpfn.data.loader import DataLoader

    dataset = DataLoader(seed=seed).load_local_csv(dataset_path, target_column="target", task_override="binary")
    stats: dict[str, Any] = {
        "n_total": int(dataset.metadata["total_samples"]),
        "n_train": int(dataset.metadata["train_samples"]),
        "n_val": int(dataset.metadata["val_samples"]),
        "n_test": int(dataset.metadata["test_samples"]),
        "class_counts_total": class_counts(pd.concat([dataset.y_train, dataset.y_val, dataset.y_test])),
        "class_counts_train": class_counts(dataset.y_train),
        "class_counts_val": class_counts(dataset.y_val),
        "class_counts_test": class_counts(dataset.y_test),
        "test_indices": [int(idx) for idx in dataset.y_test.index],
    }
    for split_name, values in {
        "train": dataset.y_train,
        "val": dataset.y_val,
        "test": dataset.y_test,
    }.items():
        counts = class_counts(values)
        stats[f"class_0_{split_name}"] = int(counts.get("0", 0))
        stats[f"class_1_{split_name}"] = int(counts.get("1", 0))

    if metadata_path and Path(metadata_path).exists():
        metadata = pd.read_csv(metadata_path)
        if "cancer_type" in metadata.columns:
            stats["cancer_counts_total"] = {str(k): int(v) for k, v in metadata["cancer_type"].value_counts().to_dict().items()}
            test_metadata = metadata.iloc[dataset.y_test.index].reset_index(drop=True)
            stats["test_cancer_counts"] = {str(k): int(v) for k, v in test_metadata["cancer_type"].value_counts().to_dict().items()}
    return stats


def _probability_column(predictions: pd.DataFrame) -> str | None:
    if "prob_1" in predictions.columns:
        return "prob_1"
    prob_cols = [col for col in predictions.columns if col.startswith("prob_")]
    if len(prob_cols) >= 2:
        return prob_cols[1]
    if len(prob_cols) == 1:
        return prob_cols[0]
    return None


def _encoded_vector(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).map({"0": 0, "1": 1}).astype(int).to_numpy()


def _extra_prediction_metrics(prediction_path: Path) -> dict[str, float | None]:
    predictions = pd.read_csv(prediction_path)
    y_true_col = "y_true_encoded" if "y_true_encoded" in predictions.columns else "y_true"
    y_pred_col = "y_pred_encoded" if "y_pred_encoded" in predictions.columns else "y_pred"
    y_true = _encoded_vector(predictions[y_true_col])
    y_pred = _encoded_vector(predictions[y_pred_col])
    prob_col = _probability_column(predictions)
    y_score = predictions[prob_col].to_numpy(dtype=float) if prob_col else None

    result: dict[str, float | None] = {
        "pr_auc": None,
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        result["pr_auc"] = float(average_precision_score(y_true, y_score))
    return result


def _subgroup_metrics(
    *,
    prediction_path: Path,
    metadata_path: str | None,
    test_indices: list[int],
    model_name: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    if not metadata_path or not Path(metadata_path).exists():
        return []
    metadata = pd.read_csv(metadata_path)
    if "cancer_type" not in metadata.columns:
        return []
    predictions = pd.read_csv(prediction_path)
    if len(test_indices) != len(predictions):
        return []

    y_true_col = "y_true_encoded" if "y_true_encoded" in predictions.columns else "y_true"
    y_pred_col = "y_pred_encoded" if "y_pred_encoded" in predictions.columns else "y_pred"
    prob_col = _probability_column(predictions)
    test_metadata = metadata.iloc[test_indices].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "cancer_type": test_metadata["cancer_type"].astype(str),
            "y_true": _encoded_vector(predictions[y_true_col]),
            "y_pred": _encoded_vector(predictions[y_pred_col]),
        }
    )
    if prob_col:
        frame["y_score"] = predictions[prob_col].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for cancer, group in frame.groupby("cancer_type"):
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "model_name": model_name,
            "cancer_type": cancer,
            "n_test": int(len(group)),
            "class_0_test": int((group["y_true"] == 0).sum()),
            "class_1_test": int((group["y_true"] == 1).sum()),
            "sensitivity": float(recall_score(group["y_true"], group["y_pred"], pos_label=1, zero_division=0)),
            "specificity": float(recall_score(group["y_true"], group["y_pred"], pos_label=0, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(group["y_true"], group["y_pred"])),
        }
        if "y_score" in group.columns and len(np.unique(group["y_true"])) == 2:
            row["pr_auc"] = float(average_precision_score(group["y_true"], group["y_score"]))
        else:
            row["pr_auc"] = None
        rows.append(row)
    return rows


def _dataset_identity(record: dict[str, Any]) -> dict[str, Any]:
    name = str(record["name"])
    endpoint = str(record.get("endpoint", ""))
    horizon_days = record.get("horizon_days")
    cancers = record.get("source_cancers") or []
    return {
        "dataset": name,
        "endpoint": endpoint,
        "horizon_days": horizon_days,
        "source_cancers": ",".join(map(str, cancers)),
        "view": record.get("view"),
        "label_rule": record.get("label_rule"),
        "dataset_path": record.get("path"),
        "metadata_path": record.get("metadata_path"),
    }


def _run_tabpfn(
    *,
    records: list[dict[str, Any]],
    output_root: Path,
    package_src: Path,
    seed: int,
    train_rows_cap: int | None,
    versions: list[str],
    legacy_v1_root: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _ensure_package_import(package_src)
    from ev_tabpfn.evaluation.generations import compare_tabpfn_generations

    dataset_items = [
        {
            "name": record["name"],
            "path": record["path"],
            "target_column": record.get("target", "target"),
            "task": "binary",
        }
        for record in records
    ]
    result = compare_tabpfn_generations(
        dataset_items,
        versions=versions,
        output_root=output_root / "tabpfn",
        target_column="target",
        task="binary",
        seed=seed,
        train_rows_cap=train_rows_cap,
        legacy_v1_root=legacy_v1_root,
    )
    summary = json.loads(Path(result.summary_path).read_text())
    by_name = {record["name"]: record for record in records}
    rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []

    for run in summary.get("runs", []):
        if run.get("status") != "success":
            rows.append({"dataset": run.get("dataset"), "status": "failed", "error_type": run.get("error_type"), "error_message": run.get("error_message")})
            continue
        dataset_name = str(run["dataset"])
        record = by_name[dataset_name]
        split = _split_stats(record["path"], record.get("metadata_path"), package_src, seed)
        run_dir = Path(run["run_dir"])
        augmented_rows: list[dict[str, Any]] = []
        for metric_row in run.get("rows", []):
            model_name = str(metric_row.get("model_name"))
            prediction_path = run_dir / "predictions" / f"{model_name}_predictions.csv"
            extra = _extra_prediction_metrics(prediction_path) if prediction_path.exists() else {}
            row = {
                **_dataset_identity(record),
                **{key: value for key, value in split.items() if key != "test_indices"},
                **metric_row,
                **extra,
                "run_dir": str(run_dir),
                "prediction_path": str(prediction_path) if prediction_path.exists() else "",
                "model_family": "tabpfn",
            }
            rows.append(row)
            augmented_rows.append(row)
            if prediction_path.exists():
                subgroup_rows.extend(
                    _subgroup_metrics(
                        prediction_path=prediction_path,
                        metadata_path=record.get("metadata_path"),
                        test_indices=split["test_indices"],
                        model_name=model_name,
                        dataset_name=dataset_name,
                    )
                )
        if augmented_rows:
            pd.DataFrame(augmented_rows).to_csv(run_dir / "metrics" / "augmented_metrics.csv", index=False)
    return rows, subgroup_rows


def _run_tabfm_one(
    *,
    record: dict[str, Any],
    output_root: Path,
    package_src: Path,
    seed: int,
    train_rows_cap: int | None,
    backend: str,
    ensemble: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ensure_package_import(package_src)
    from ev_tabpfn.artifacts.paths import create_run_dirs
    from ev_tabpfn.artifacts.writers import setup_logger, write_json, write_metrics_csv, write_predictions, write_raw_predictions
    from ev_tabpfn.data.loader import DataLoader
    from ev_tabpfn.evaluation.labels import ClassificationLabelContract
    from ev_tabpfn.evaluation.metrics import classification_metrics
    from ev_tabpfn.models.tabfm_backend import TabFMAdapter
    from ev_tabpfn.reporting.plots import save_classification_plots

    model_name = "tabfm_ensemble" if ensemble else "tabfm_default"
    dirs = create_run_dirs(output_root / "tabfm", record["name"])
    logger = setup_logger(dirs["logs"] / "run.log", logger_name="cancer_os.tabfm")
    dataset = DataLoader(seed=seed).load_local_csv(record["path"], target_column=record.get("target", "target"), task_override="binary")
    label_contract = ClassificationLabelContract.from_labels(dataset.task_type, dataset.y_train)
    split = _split_stats(record["path"], record.get("metadata_path"), package_src, seed)

    write_json(
        dirs["metadata"] / "dataset_metadata.json",
        {
            "dataset": record["name"],
            "dataset_path": record["path"],
            "metadata_path": record.get("metadata_path"),
            "task_type": dataset.task_type,
            "target": record.get("target", "target"),
            "seed": seed,
            "train_rows_cap": train_rows_cap,
            "tabfm_backend": backend,
            "tabfm_ensemble": ensemble,
            "split_stats": {key: value for key, value in split.items() if key != "test_indices"},
        },
    )

    try:
        adapter = TabFMAdapter(
            task_type="binary",
            backend=backend,
            ensemble=ensemble,
            max_train_rows=train_rows_cap,
            random_state=seed,
        )
        logger.info("Fitting %s on %s", model_name, record["name"])
        adapter.fit(dataset.X_train, dataset.y_train)
        y_prob = np.asarray(adapter.predict_proba(dataset.X_test))
        y_pred = np.asarray(adapter.predict(dataset.X_test))
        metrics = classification_metrics(
            "binary",
            dataset.y_test,
            y_pred,
            y_prob,
            label_contract=label_contract,
            y_prob_classes=getattr(adapter, "classes_", None),
        )
        write_predictions(
            dirs["predictions"] / f"{model_name}_predictions.csv",
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            label_contract=label_contract,
            y_prob_classes=getattr(adapter, "classes_", None),
        )
        raw_paths = write_raw_predictions(
            dirs["base"],
            model_name=model_name,
            dataset_name=record["name"],
            task_type="binary",
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            y_prob_classes=getattr(adapter, "classes_", None),
        )
        plot_files = save_classification_plots(
            output_dir=dirs["plots"],
            task_type="binary",
            model_name=model_name,
            y_true=dataset.y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            label_contract=label_contract,
            y_prob_classes=getattr(adapter, "classes_", None),
        )
        prediction_path = dirs["predictions"] / f"{model_name}_predictions.csv"
        extra = _extra_prediction_metrics(prediction_path)
        row = {
            **_dataset_identity(record),
            **{key: value for key, value in split.items() if key != "test_indices"},
            "model_name": model_name,
            "model_family": "tabfm",
            "task_type": "binary",
            "status": "success",
            **metrics,
            **extra,
            "run_dir": str(dirs["base"]),
            "prediction_path": str(prediction_path),
            "raw_paths": json.dumps(raw_paths),
            "plot_files": json.dumps(plot_files),
        }
        write_metrics_csv(dirs["metrics"] / "metrics_summary.csv", [row])
        write_json(dirs["metadata"] / "model_status.json", {model_name: {"status": "success", "raw_paths": raw_paths, "plot_files": plot_files}})
        subgroup_rows = _subgroup_metrics(
            prediction_path=prediction_path,
            metadata_path=record.get("metadata_path"),
            test_indices=split["test_indices"],
            model_name=model_name,
            dataset_name=record["name"],
        )
        return row, subgroup_rows
    except Exception as exc:
        traceback_path = dirs["logs"] / f"{model_name}_traceback.txt"
        traceback_path.write_text(traceback.format_exc())
        row = {
            **_dataset_identity(record),
            **{key: value for key, value in split.items() if key != "test_indices"},
            "model_name": model_name,
            "model_family": "tabfm",
            "task_type": "binary",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_path": str(traceback_path),
            "run_dir": str(dirs["base"]),
        }
        write_metrics_csv(dirs["metrics"] / "metrics_summary.csv", [row])
        write_json(dirs["metadata"] / "model_status.json", {model_name: {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)}})
        return row, []


def _write_aggregate(output_root: Path, rows: list[dict[str, Any]], subgroup_rows: list[dict[str, Any]], run_config: dict[str, Any]) -> None:
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(aggregate / "all_model_metrics.csv", index=False)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(aggregate / "subgroup_metrics_by_cancer.csv", index=False)
    else:
        pd.DataFrame().to_csv(aggregate / "subgroup_metrics_by_cancer.csv", index=False)

    metric_cols = ["accuracy", "f1", "roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "log_loss"]
    df = pd.DataFrame(rows)
    if not df.empty and "status" in df.columns:
        successful = df[df["status"] == "success"].copy()
        available = [col for col in metric_cols if col in successful.columns]
        if available:
            successful.groupby("model_name", as_index=False)[available].mean().to_csv(aggregate / "mean_metrics_by_model.csv", index=False)
            if "endpoint" in successful.columns:
                successful.groupby(["endpoint", "model_name"], as_index=False)[available].mean().to_csv(
                    aggregate / "mean_metrics_by_endpoint_model.csv",
                    index=False,
                )
    (aggregate / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))


def run_from_manifest(
    *,
    manifest_path: Path,
    output_root: Path,
    package_src: Path,
    seed: int,
    train_rows_cap: int | None,
    models: list[str],
    tabpfn_versions: list[str],
    tabfm_backend: str,
    tabfm_ensemble: bool,
    legacy_v1_root: str | None,
) -> dict[str, Any]:
    records = _read_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []

    if "tabpfn" in models:
        tabpfn_rows, tabpfn_subgroups = _run_tabpfn(
            records=records,
            output_root=output_root,
            package_src=package_src,
            seed=seed,
            train_rows_cap=train_rows_cap,
            versions=tabpfn_versions,
            legacy_v1_root=legacy_v1_root,
        )
        rows.extend(tabpfn_rows)
        subgroup_rows.extend(tabpfn_subgroups)

    if "tabfm" in models:
        for record in records:
            row, groups = _run_tabfm_one(
                record=record,
                output_root=output_root,
                package_src=package_src,
                seed=seed,
                train_rows_cap=train_rows_cap,
                backend=tabfm_backend,
                ensemble=tabfm_ensemble,
            )
            rows.append(row)
            subgroup_rows.extend(groups)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "package_src": str(package_src),
        "seed": seed,
        "train_rows_cap": train_rows_cap,
        "models": models,
        "tabpfn_versions": tabpfn_versions,
        "tabfm_backend": tabfm_backend,
        "tabfm_ensemble": tabfm_ensemble,
        "legacy_v1_root": legacy_v1_root,
    }
    _write_aggregate(output_root, rows, subgroup_rows, run_config)
    return {"rows": len(rows), "subgroup_rows": len(subgroup_rows), "output_root": str(output_root)}


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TabPFN/TabFM foundation-model survival classification experiments from an exported manifest.")
    parser.add_argument("--manifest", required=True, help="Path to an export manifest.json file.")
    parser.add_argument("--output-root", required=True, help="Folder where raw predictions, metrics, plots, and aggregate tables are written.")
    parser.add_argument("--package-src", default=str(DEFAULT_PACKAGE_SRC), help="Path to ev-tabpfn package src directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows-cap", type=int, default=1024, help="Maximum training rows passed to foundation models. Use -1 for no cap.")
    parser.add_argument("--models", default="tabfm,tabpfn", help="Comma-separated model families: tabfm,tabpfn.")
    parser.add_argument("--tabpfn-versions", default="v2,v2.5,v2.6,v3", help="Comma-separated TabPFN versions. v1 requires --legacy-v1-root.")
    parser.add_argument("--legacy-v1-root", default=None, help="Optional legacy TabPFN v1 source root. Keep unset unless a compatible v1 runtime is prepared.")
    parser.add_argument("--tabfm-backend", default="jax", choices=["jax", "pytorch"])
    parser.add_argument("--tabfm-ensemble", action="store_true", help="Run TabFM ensemble preset instead of the default single-pass preset.")
    args = parser.parse_args()

    train_rows_cap = None if args.train_rows_cap < 0 else args.train_rows_cap
    result = run_from_manifest(
        manifest_path=Path(args.manifest).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        package_src=Path(args.package_src).expanduser().resolve(),
        seed=args.seed,
        train_rows_cap=train_rows_cap,
        models=_csv_list(args.models),
        tabpfn_versions=_csv_list(args.tabpfn_versions),
        tabfm_backend=args.tabfm_backend,
        tabfm_ensemble=args.tabfm_ensemble,
        legacy_v1_root=args.legacy_v1_root,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
