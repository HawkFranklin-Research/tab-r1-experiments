"""Paired within-cancer comparison of pooled-trained and cancer-specific models.

For every patient, out-of-fold test probabilities are averaged across the five
repeats, separately for the model trained on that patient's cancer alone and the
model trained on the randomly pooled cohort. Both are then scored *within* each
cancer, so between-cancer differences in outcome prevalence cannot contribute.
Uncertainty comes from a class-stratified patient bootstrap. Reads saved
predictions only; no model is refitted.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "12")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PAPER = Path(__file__).resolve().parents[1]
LOCAL_ROOT = RESULTS / "cancer/local_models"
CLOUD_ROOT = RESULTS / "cancer/cloud_models/tabr1_results"
SOURCE_ROOT = RESULTS / "source_data/manuscript"
METRICS_PATH = SOURCE_ROOT / "model_fold_metrics.csv"

N_BOOT = 2000
SEED = 20260901


def prediction_path(row: pd.Series) -> Path:
    base = LOCAL_ROOT if row["artifact_source"] == "local" else CLOUD_ROOT
    return (
        base / row["scope"] / row["endpoint"] / row["cancer"]
        / f"repeat_{int(row['repeat']):02d}_fold_{int(row['fold']):02d}"
        / row["model_name"] / "test_predictions.csv"
    )


def load_patient_probabilities(metrics: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in metrics[metrics["roc_auc"].notna()].iterrows():
        path = prediction_path(row)
        if not path.exists():
            continue
        pred = pd.read_csv(path, usecols=["patient_id", "cancer_type", "y_true", "probability"])
        pred["scope"] = row["scope"]
        pred["endpoint"] = row["endpoint"]
        pred["model_name"] = row["model_name"]
        frames.append(pred)
    predictions = pd.concat(frames, ignore_index=True)
    return (
        predictions.groupby(["scope", "endpoint", "model_name", "cancer_type", "patient_id", "y_true"], as_index=False)
        ["probability"].mean()
    )


def auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])


def main() -> None:
    metrics = pd.read_csv(METRICS_PATH)
    probs = load_patient_probabilities(metrics)
    wide = probs.pivot_table(
        index=["endpoint", "model_name", "cancer_type", "patient_id", "y_true"],
        columns="scope", values="probability",
    ).dropna(subset=["per_cancer", "pooled"]).reset_index()

    rng = np.random.default_rng(SEED)
    rows, summary_rows = [], []
    for (endpoint, model), task in wide.groupby(["endpoint", "model_name"]):
        boot_deltas = []
        for cancer, group in task.groupby("cancer_type"):
            y = group["y_true"].to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            p_cancer, p_pool = group["per_cancer"].to_numpy(), group["pooled"].to_numpy()
            draws = np.empty((N_BOOT, 2))
            for b in range(N_BOOT):
                idx = bootstrap_indices(y, rng)
                draws[b] = auc(y[idx], p_cancer[idx]), auc(y[idx], p_pool[idx])
            delta = draws[:, 1] - draws[:, 0]
            boot_deltas.append(delta)
            rows.append({
                "endpoint": endpoint, "model_name": model, "cancer": cancer,
                "n_patients": len(y), "n_events": int(y.sum()),
                "auc_cancer_specific": auc(y, p_cancer), "auc_pooled": auc(y, p_pool),
                "delta_pooled_minus_specific": auc(y, p_pool) - auc(y, p_cancer),
                "delta_ci_low": np.percentile(delta, 2.5), "delta_ci_high": np.percentile(delta, 97.5),
                "p_pooled_better": float((delta > 0).mean()),
            })
        if boot_deltas:
            mean_delta = np.mean(boot_deltas, axis=0)
            task_rows = [r for r in rows if r["endpoint"] == endpoint and r["model_name"] == model]
            summary_rows.append({
                "endpoint": endpoint, "model_name": model, "n_cancers": len(task_rows),
                "n_patients": sum(r["n_patients"] for r in task_rows),
                "mean_auc_cancer_specific": np.mean([r["auc_cancer_specific"] for r in task_rows]),
                "mean_auc_pooled": np.mean([r["auc_pooled"] for r in task_rows]),
                "mean_delta": np.mean([r["delta_pooled_minus_specific"] for r in task_rows]),
                "mean_delta_ci_low": np.percentile(mean_delta, 2.5),
                "mean_delta_ci_high": np.percentile(mean_delta, 97.5),
            })

    per_cancer = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    per_cancer.to_csv(SOURCE_ROOT / "figure_03_within_cancer_pooling_contrast.csv", index=False)
    summary.to_csv(SOURCE_ROOT / "figure_03_within_cancer_pooling_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
