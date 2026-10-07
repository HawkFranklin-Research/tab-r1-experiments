from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ev_tabpfn.evaluation.labels import ClassificationLabelContract, flatten_predictions


def setup_logger(log_path: Path, logger_name: str = "ev_tabpfn") -> logging.Logger:
    logger = logging.getLogger(f"{logger_name}.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_predictions(
    path: Path,
    *,
    y_true: pd.Series,
    y_pred: Any,
    y_prob: Any,
    label_contract: ClassificationLabelContract | None = None,
    y_prob_classes: Any | None = None,
) -> None:
    y_pred_flat = flatten_predictions(y_pred)
    df = pd.DataFrame({"y_true": y_true.reset_index(drop=True), "y_pred": pd.Series(y_pred_flat)})
    if label_contract is not None:
        df["y_true_encoded"] = label_contract.encode(y_true)
        df["y_pred_encoded"] = label_contract.encode(y_pred_flat)
    if y_prob is not None:
        if label_contract is not None:
            prob_df = label_contract.probability_frame(y_prob, y_prob_classes)
        else:
            prob_df = pd.DataFrame(y_prob).add_prefix("prob_")
        df = pd.concat([df, prob_df], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_raw_predictions(
    output_dir: Path,
    *,
    model_name: str,
    dataset_name: str,
    task_type: str,
    y_true: Any,
    y_pred: Any,
    y_prob: Any = None,
    y_prob_classes: Any | None = None,
) -> dict[str, str]:
    raw_dir = output_dir / "raw" / model_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "task_type": task_type,
        "y_true": np.asarray(y_true).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_prob": None if y_prob is None else np.asarray(y_prob).tolist(),
        "y_prob_classes": None if y_prob_classes is None else np.asarray(y_prob_classes).tolist(),
    }
    json_path = raw_dir / "raw_predictions.json"
    npz_path = raw_dir / "raw_predictions.npz"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    np.savez_compressed(
        npz_path,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_prob=np.asarray(y_prob) if y_prob is not None else np.array([]),
    )
    return {"json": str(json_path), "npz": str(npz_path)}


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
