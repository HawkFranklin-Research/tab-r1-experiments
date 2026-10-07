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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold

from resource_limits import configure_process_limits


ROOT = REPO
DEFAULT_INPUT_ROOT = TRAIN_READY
DEFAULT_OUTPUT_ROOT = FROZEN
HORIZONS = {"os_3yr": 1095, "os_5yr": 1825}


@dataclass
class CohortMatrix:
    cancer: str
    X: sparse.csr_matrix
    metadata: pd.DataFrame
    feature_ids: np.ndarray
    y: np.ndarray
    endpoint: str
    excluded: int


def endpoint_labels(metadata: pd.DataFrame, endpoint: str) -> tuple[np.ndarray, np.ndarray, int]:
    days = pd.to_numeric(metadata["OS_days"], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(metadata["OS_event"], errors="coerce").to_numpy(dtype=float)
    known = np.isfinite(days) & np.isfinite(event)
    if endpoint in HORIZONS:
        horizon = HORIZONS[endpoint]
        positive = known & (event == 1) & (days <= horizon)
        negative = known & (((event == 0) & (days >= horizon)) | ((event == 1) & (days > horizon)))
    elif endpoint == "extreme_os":
        positive = known & (event == 1) & (days < HORIZONS["os_3yr"])
        negative = known & (event == 0) & (days >= HORIZONS["os_5yr"])
    else:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    usable = positive | negative
    labels = positive[usable].astype(np.int8)
    return usable, labels, int((known & ~usable).sum())


def load_cohort(input_root: Path, cancer: str, view: str, endpoint: str) -> CohortMatrix:
    base = input_root / cancer / view
    X = sparse.load_npz(base / "X.npz").tocsr().astype(np.float32)
    metadata = pd.read_csv(base / "sample_index.csv")
    features = pd.read_csv(base / "feature_index.csv")
    if X.shape != (len(metadata), len(features)):
        raise ValueError(
            f"{cancer}/{view} contract mismatch: X={X.shape}, metadata={len(metadata)}, features={len(features)}"
        )
    usable, y, excluded = endpoint_labels(metadata, endpoint)
    metadata = metadata.loc[usable].reset_index(drop=True).copy()
    metadata["cancer_type"] = cancer
    metadata["target"] = y
    feature_ids = features["feature_id"].astype(str).to_numpy()
    nonclinical = ~np.char.startswith(feature_ids.astype(str), "clinical::")
    return CohortMatrix(
        cancer=cancer,
        X=X[usable][:, nonclinical].tocsr(),
        metadata=metadata,
        feature_ids=feature_ids[nonclinical],
        y=y,
        endpoint=endpoint,
        excluded=excluded,
    )


def patient_groups(metadata: pd.DataFrame) -> np.ndarray:
    if "patient_id" in metadata.columns:
        groups = metadata["patient_id"].fillna(metadata.get("sample_id")).astype(str)
    elif "sample_id" in metadata.columns:
        groups = metadata["sample_id"].astype(str)
    else:
        groups = pd.Series(np.arange(len(metadata)), dtype=str)
    return groups.to_numpy()


def feature_modality(feature_id: str) -> str:
    for separator in ("::", "__", ":"):
        if separator in feature_id:
            return feature_id.split(separator, 1)[0]
    return "unknown"


def variance_rank(X_train: sparse.csr_matrix, max_features: int) -> np.ndarray:
    mean = np.asarray(X_train.mean(axis=0)).ravel()
    mean_sq = np.asarray(X_train.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0.0)
    count = min(max_features, X_train.shape[1])
    if count == X_train.shape[1]:
        return np.argsort(variance)[::-1]
    candidates = np.argpartition(variance, -count)[-count:]
    return candidates[np.argsort(variance[candidates])[::-1]]


def harmonize_cohorts(cohorts: list[CohortMatrix]) -> CohortMatrix:
    common = set(cohorts[0].feature_ids.tolist())
    for cohort in cohorts[1:]:
        common.intersection_update(cohort.feature_ids.tolist())
    common_ids = np.asarray(sorted(common), dtype=str)
    if not len(common_ids):
        raise ValueError("No common non-clinical features across selected cancers")

    matrices: list[sparse.csr_matrix] = []
    metadata: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for cohort in cohorts:
        positions = {feature_id: position for position, feature_id in enumerate(cohort.feature_ids)}
        aligned_positions = np.fromiter((positions[item] for item in common_ids), dtype=np.int64)
        matrices.append(cohort.X[:, aligned_positions])
        metadata.append(cohort.metadata)
        labels.append(cohort.y)
    endpoint = cohorts[0].endpoint
    return CohortMatrix(
        cancer="ALL",
        X=sparse.vstack(matrices, format="csr", dtype=np.float32),
        metadata=pd.concat(metadata, ignore_index=True),
        feature_ids=common_ids,
        y=np.concatenate(labels),
        endpoint=endpoint,
        excluded=sum(cohort.excluded for cohort in cohorts),
    )


def inner_validation_split(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    subset_y = y[indices]
    subset_groups = groups[indices]
    class_group_counts = [len(np.unique(subset_groups[subset_y == label])) for label in np.unique(subset_y)]
    n_splits = min(5, min(class_group_counts))
    if n_splits < 2:
        raise ValueError("Not enough patient groups per class for a grouped validation split")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_local, val_local = next(splitter.split(np.zeros(len(indices)), subset_y, subset_groups))
    return indices[train_local], indices[val_local]


def safe_feature_name(feature_id: str) -> str:
    return feature_id.replace("::", "__").replace(":", "_").replace("/", "_").replace(" ", "_")


def write_split(
    directory: Path,
    split_name: str,
    cohort: CohortMatrix,
    row_indices: np.ndarray,
    feature_positions: np.ndarray,
) -> tuple[str, str]:
    feature_names = [safe_feature_name(value) for value in cohort.feature_ids[feature_positions]]
    frame = pd.DataFrame(
        cohort.X[row_indices][:, feature_positions].toarray(),
        columns=feature_names,
    )
    frame["target"] = cohort.y[row_indices]
    data_path = directory / f"{split_name}.csv"
    metadata_path = directory / f"{split_name}_metadata.csv"
    frame.to_csv(data_path, index=False)
    cohort.metadata.iloc[row_indices].reset_index(drop=True).to_csv(metadata_path, index=False)
    return str(data_path), str(metadata_path)


def grouped_folds(
    cohort: CohortMatrix,
    *,
    repeats: int,
    outer_splits: int,
    seed: int,
) -> Iterable[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    groups = patient_groups(cohort.metadata)
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=outer_splits,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (development, test) in enumerate(splitter.split(cohort.X, cohort.y, groups)):
            train, validation = inner_validation_split(
                development,
                cohort.y,
                groups,
                seed=seed + repeat * 100 + fold,
            )
            yield repeat, fold, train, validation, test


def export_cohort_folds(
    cohort: CohortMatrix,
    *,
    scope: str,
    output_root: Path,
    repeats: int,
    outer_splits: int,
    max_features: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for repeat, fold, train, validation, test in grouped_folds(
        cohort,
        repeats=repeats,
        outer_splits=outer_splits,
        seed=seed,
    ):
        selected = variance_rank(cohort.X[train], max_features=max_features)
        directory = output_root / scope / cohort.endpoint / cohort.cancer / f"repeat_{repeat:02d}_fold_{fold:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for split_name, indices in (("train", train), ("validation", validation), ("test", test)):
            data_path, metadata_path = write_split(directory, split_name, cohort, indices, selected)
            paths[f"{split_name}_path"] = data_path
            paths[f"{split_name}_metadata_path"] = metadata_path

        selected_features = pd.DataFrame(
            {
                "rank": np.arange(1, len(selected) + 1),
                "feature_id": cohort.feature_ids[selected],
                "modality": [feature_modality(item) for item in cohort.feature_ids[selected]],
            }
        )
        feature_path = directory / "selected_features.csv"
        selected_features.to_csv(feature_path, index=False)
        record = {
            "scope": scope,
            "cancer": cohort.cancer,
            "endpoint": cohort.endpoint,
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "n_total": len(cohort.y),
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "class_1_train": int(cohort.y[train].sum()),
            "class_1_validation": int(cohort.y[validation].sum()),
            "class_1_test": int(cohort.y[test].sum()),
            "patient_overlap_train_test": len(
                set(patient_groups(cohort.metadata)[train]) & set(patient_groups(cohort.metadata)[test])
            ),
            "feature_candidates": cohort.X.shape[1],
            "features_selected": len(selected),
            "feature_selection": "variance ranking fitted on training rows only",
            "selected_features_path": str(feature_path),
            **paths,
        }
        (directory / "fold_config.json").write_text(json.dumps(record, indent=2))
        records.append(record)
    return records


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export patient-grouped, leakage-safe cancer OS folds.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cancers", default="BRCA,ESCA,HNSCC,LSCC,LUAD")
    parser.add_argument("--endpoints", default="os_3yr,os_5yr,extreme_os")
    parser.add_argument("--scopes", default="per_cancer,pooled")
    parser.add_argument("--view", default="core", choices=["core", "proteogenomic"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)

    cancers = parse_csv_list(args.cancers)
    endpoints = parse_csv_list(args.endpoints)
    scopes = parse_csv_list(args.scopes)
    if args.smoke:
        cancers = ["ESCA"]
        endpoints = ["os_3yr"]
        scopes = ["per_cancer"]
        args.repeats = 1
        args.outer_splits = 2
        args.max_features = min(args.max_features, 10)

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    all_records: list[dict[str, Any]] = []
    for endpoint in endpoints:
        cohorts = [load_cohort(input_root, cancer, args.view, endpoint) for cancer in cancers]
        cohorts = [
            cohort
            for cohort in cohorts
            if len(cohort.y) >= 2 * args.min_class_count
            and np.bincount(cohort.y, minlength=2).min() >= args.min_class_count
        ]
        if "per_cancer" in scopes:
            for cohort in cohorts:
                all_records.extend(
                    export_cohort_folds(
                        cohort,
                        scope="per_cancer",
                        output_root=output_root,
                        repeats=args.repeats,
                        outer_splits=args.outer_splits,
                        max_features=args.max_features,
                        seed=args.seed,
                    )
                )
        if "pooled" in scopes and len(cohorts) >= 2:
            pooled = harmonize_cohorts(cohorts)
            all_records.extend(
                export_cohort_folds(
                    pooled,
                    scope="pooled",
                    output_root=output_root,
                    repeats=args.repeats,
                    outer_splits=args.outer_splits,
                    max_features=args.max_features,
                    seed=args.seed,
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(output_root / "fold_manifest.csv", index=False)
    (output_root / "fold_manifest.json").write_text(
        json.dumps(
            {
                "records": all_records,
                "input_root": str(input_root),
                "view": args.view,
                "cancers": cancers,
                "endpoints": endpoints,
                "scopes": scopes,
                "repeats": args.repeats,
                "outer_splits": args.outer_splits,
                "max_features": args.max_features,
                "patient_grouped": True,
                "feature_selection_within_training_fold": True,
                "smoke": args.smoke,
                "models_executed": False,
            },
            indent=2,
        )
    )
    print(f"Exported {len(all_records)} fold records to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
