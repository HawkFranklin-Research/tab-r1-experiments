from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .labels import ClassificationLabelContract, flatten_predictions


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def create_run_dirs(output_root: str | Path, dataset_name: str) -> dict[str, Path]:
    root = Path(output_root)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / "runs" / slugify(dataset_name) / run_id
    dirs = {
        "base": base,
        "predictions": base / "predictions",
        "metrics": base / "metrics",
        "plots": base / "plots",
        "metadata": base / "metadata",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"phase2.{log_path.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
            prob_df = pd.DataFrame(y_prob)
            prob_df = prob_df.add_prefix("prob_")
        df = pd.concat([df, prob_df], axis=1)
    df.to_csv(path, index=False)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
