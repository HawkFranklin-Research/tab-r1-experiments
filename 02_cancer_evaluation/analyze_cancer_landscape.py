from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from resource_limits import configure_process_limits, thread_limit


ROOT = REPO
RAW_ROOT = TRAIN_READY
CANCER_ROOT = REPO / "02_cancer_evaluation/historical_split"
CANCERS = ("BRCA", "ESCA", "HNSCC", "LSCC", "LUAD")


def read_manifest(experiment: str) -> list[dict[str, Any]]:
    path = CANCER_ROOT / experiment / "datasets" / "manifest.json"
    return json.loads(path.read_text())["records"]


def cohort_counts(cancers: list[str]) -> pd.DataFrame:
    records = read_manifest("exp01_per_cancer_fixed_window")
    rows: list[dict[str, Any]] = []
    for cancer in cancers:
        metadata = pd.read_csv(RAW_ROOT / cancer / "core" / "sample_index.csv")
        row: dict[str, Any] = {
            "cancer": cancer,
            "n_raw": len(metadata),
            "n_patients": metadata.get("patient_id", metadata.get("sample_id")).nunique(),
        }
        for record in records:
            if record["source_cancers"] == [cancer]:
                row[f"n_{record['endpoint']}"] = record["samples"]
                row[f"excluded_{record['endpoint']}"] = record.get("excluded_ambiguous")
        rows.append(row)
    return pd.DataFrame(rows)


def class_balance(cancers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in read_manifest("exp01_per_cancer_fixed_window"):
        cancer = record["source_cancers"][0]
        if cancer not in cancers:
            continue
        for label, count in record["class_counts"].items():
            rows.append(
                {
                    "cancer": cancer,
                    "endpoint": record["endpoint"],
                    "class": int(label),
                    "count": int(count),
                    "n_total": int(record["samples"]),
                }
            )
    return pd.DataFrame(rows)


def kaplan_meier(metadata: pd.DataFrame, cancer: str) -> pd.DataFrame:
    days = pd.to_numeric(metadata["OS_days"], errors="coerce")
    event = pd.to_numeric(metadata["OS_event"], errors="coerce")
    valid = days.notna() & event.isin([0, 1]) & days.ge(0)
    frame = pd.DataFrame({"time": days[valid], "event": event[valid].astype(int)}).sort_values("time")
    event_times = np.sort(frame.loc[frame["event"].eq(1), "time"].unique())
    survival = 1.0
    greenwood = 0.0
    rows = [{"cancer": cancer, "time_days": 0.0, "survival": 1.0, "ci_low": 1.0, "ci_high": 1.0}]
    for time_value in event_times:
        at_risk = int((frame["time"] >= time_value).sum())
        events = int(((frame["time"] == time_value) & frame["event"].eq(1)).sum())
        if at_risk <= events:
            survival = 0.0
        else:
            survival *= 1.0 - events / at_risk
            greenwood += events / (at_risk * (at_risk - events))
        standard_error = survival * np.sqrt(greenwood) if survival > 0 else 0.0
        rows.append(
            {
                "cancer": cancer,
                "time_days": float(time_value),
                "survival": float(survival),
                "ci_low": float(max(0.0, survival - 1.96 * standard_error)),
                "ci_high": float(min(1.0, survival + 1.96 * standard_error)),
            }
        )
    return pd.DataFrame(rows)


def selected_feature_overlap(cancers: list[str], endpoint: str = "os_3yr") -> pd.DataFrame:
    feature_sets: dict[str, set[str]] = {}
    for record in read_manifest("exp01_per_cancer_fixed_window"):
        cancer = record["source_cancers"][0]
        if cancer not in cancers or record["endpoint"] != endpoint:
            continue
        path = Path(record["path"])
        if not path.exists():
            path = CANCER_ROOT / "exp01_per_cancer_fixed_window" / "datasets" / path.name
        columns = pd.read_csv(path, nrows=0).columns
        feature_sets[cancer] = set(columns) - {"target"}
    rows: list[dict[str, Any]] = []
    for left in cancers:
        for right in cancers:
            left_set = feature_sets.get(left, set())
            right_set = feature_sets.get(right, set())
            union = left_set | right_set
            rows.append(
                {
                    "cancer_a": left,
                    "cancer_b": right,
                    "intersection": len(left_set & right_set),
                    "union": len(union),
                    "jaccard": len(left_set & right_set) / len(union) if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cohort_separability(
    *, folds: int, seed: int, threads: int, max_rows: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    record = read_manifest("exp02_combined_fixed_window")[0]
    data_path = Path(record["path"])
    metadata_path = Path(record["metadata_path"])
    if not data_path.exists():
        data_path = CANCER_ROOT / "exp02_combined_fixed_window" / "datasets" / data_path.name
    if not metadata_path.exists():
        metadata_path = CANCER_ROOT / "exp02_combined_fixed_window" / "datasets" / metadata_path.name
    data = pd.read_csv(data_path).drop(columns=["target"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    metadata = pd.read_csv(metadata_path)
    if max_rows is not None and len(data) > max_rows:
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        for _, positions in metadata.groupby("cancer_type").groups.items():
            positions = np.asarray(list(positions), dtype=int)
            allocation = max(2, round(max_rows * len(positions) / len(data)))
            selected.extend(rng.choice(positions, size=min(allocation, len(positions)), replace=False).tolist())
        selected = sorted(selected[:max_rows])
        data = data.iloc[selected].reset_index(drop=True)
        metadata = metadata.iloc[selected].reset_index(drop=True)
    y = metadata["cancer_type"].astype(str).to_numpy()
    feature_sets = {
        "molecular_values": data,
        "structural_zero_pattern": data.eq(0).astype(np.int8),
    }
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = sorted(np.unique(y))
    for control, X in feature_sets.items():
        estimator = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)
        with thread_limit(threads):
            predicted = cross_val_predict(estimator, X, y, cv=splitter, method="predict", n_jobs=threads)
        rows.append(
            {
                "control": control,
                "n": len(y),
                "folds": folds,
                "accuracy": float(np.mean(predicted == y)),
                "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            }
        )
        matrix = confusion_matrix(y, predicted, labels=labels, normalize="true")
        confusion_rows.extend(
            {
                "control": control,
                "true_cancer": true_label,
                "predicted_cancer": predicted_label,
                "fraction": float(matrix[i, j]),
            }
            for i, true_label in enumerate(labels)
            for j, predicted_label in enumerate(labels)
        )
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build machine-readable cancer landscape source tables.")
    parser.add_argument("--output-dir", default=str(ROOT / "paper" / "tables" / "source_data" / "landscape"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)
    cancers = list(CANCERS[:2]) if args.smoke else list(CANCERS)
    folds = 2 if args.smoke else 5
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = cohort_counts(cancers)
    balance = class_balance(cancers)
    km = pd.concat(
        [kaplan_meier(pd.read_csv(RAW_ROOT / cancer / "core" / "sample_index.csv"), cancer) for cancer in cancers],
        ignore_index=True,
    )
    overlap = selected_feature_overlap(cancers)
    modality = pd.read_csv(CANCER_ROOT / "shared" / "reports" / "feature_modality_selected_prefix_counts.csv")
    modality = modality.loc[modality["cancer"].isin(cancers)].reset_index(drop=True)
    separability, confusion = cohort_separability(
        folds=folds,
        seed=args.seed,
        threads=args.threads,
        max_rows=300 if args.smoke else None,
    )

    for name, frame in {
        "cohort_counts": counts,
        "class_balance": balance,
        "kaplan_meier": km,
        "selected_feature_overlap": overlap,
        "selected_feature_modalities": modality,
        "cohort_separability": separability,
        "cohort_separability_confusion": confusion,
    }.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "landscape_config.json").write_text(
        json.dumps(
            {
                "cancers": cancers,
                "separability_folds": folds,
                "seed": args.seed,
                "threads": args.threads,
                "memory_gb": args.memory_gb,
                "smoke": args.smoke,
                "foundation_models_executed": False,
            },
            indent=2,
        )
    )
    print(counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
