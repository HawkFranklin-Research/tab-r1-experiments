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
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_saved_cancer_results import (
    binary_metrics,
    fit_probability,
    load_pooled_records,
    make_logistic,
    split_record,
)
from resource_limits import configure_process_limits


ROOT = REPO


def molecular_estimator(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", make_logistic(seed)),
        ]
    )


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def evaluate_partition(
    *,
    record: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    partition_type: str,
    partition_value: str,
    seed: int,
    threads: int,
) -> list[dict[str, Any]]:
    dataset, metadata = split_record(record)
    full_X = pd.concat([dataset.X_train, dataset.X_val, dataset.X_test]).sort_index()
    full_y = pd.concat([dataset.y_train, dataset.y_val, dataset.y_test]).sort_index().to_numpy(dtype=int)
    X = numeric_features(full_X)
    y_train = full_y[train_indices]
    y_test = full_y[test_indices]
    if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
        return []

    model_inputs = {
        "linear_molecular": (X.iloc[train_indices], X.iloc[test_indices], molecular_estimator(seed)),
        "structural_zero_pattern": (
            X.iloc[train_indices].fillna(0).eq(0).astype(np.int8),
            X.iloc[test_indices].fillna(0).eq(0).astype(np.int8),
            make_logistic(seed),
        ),
    }
    rows: list[dict[str, Any]] = []
    for control, (X_train, X_test, estimator) in model_inputs.items():
        probability = fit_probability(estimator, X_train, y_train, X_test, threads=threads)
        rows.append(
            {
                "experiment": record["experiment"],
                "dataset": record["name"],
                "endpoint": record.get("endpoint"),
                "partition_type": partition_type,
                "partition_value": partition_value,
                "control": control,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "class_0_test": int((y_test == 0).sum()),
                "class_1_test": int((y_test == 1).sum()),
                **binary_metrics(y_test, probability),
                "feature_space_warning": "Features were preselected before this stress test; use the repeated-validation utility for fold-internal selection.",
            }
        )
    return rows


def held_out_tests(records: list[dict[str, Any]], *, seed: int, threads: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        _, metadata = split_record(record)
        cancer = metadata["cancer_type"].astype(str).to_numpy()
        for held_out in sorted(np.unique(cancer)):
            test_indices = np.flatnonzero(cancer == held_out)
            train_indices = np.flatnonzero(cancer != held_out)
            rows.extend(
                evaluate_partition(
                    record=record,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    partition_type="leave_one_cancer_out",
                    partition_value=held_out,
                    seed=seed,
                    threads=threads,
                )
            )

        source_column = "cohort_x" if "cohort_x" in metadata.columns else "cohort_y"
        sources = metadata[source_column].astype(str).to_numpy()
        for held_out in sorted(np.unique(sources)):
            test_indices = np.flatnonzero(sources == held_out)
            train_indices = np.flatnonzero(sources != held_out)
            rows.extend(
                evaluate_partition(
                    record=record,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    partition_type="leave_one_source_out",
                    partition_value=held_out,
                    seed=seed,
                    threads=threads,
                )
            )
    return pd.DataFrame(rows)


def permutation_tests(
    records: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
    threads: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for record in records:
        dataset, metadata = split_record(record)
        X_train = numeric_features(dataset.X_train)
        X_test = numeric_features(dataset.X_test)
        y_train = dataset.y_train.to_numpy(dtype=int)
        y_test = dataset.y_test.to_numpy(dtype=int)
        train_cancer = metadata.iloc[dataset.y_train.index]["cancer_type"].astype(str).to_numpy()

        for iteration in range(iterations):
            global_labels = rng.permutation(y_train)
            within_labels = y_train.copy()
            for cancer in np.unique(train_cancer):
                positions = np.flatnonzero(train_cancer == cancer)
                within_labels[positions] = rng.permutation(within_labels[positions])

            for permutation, labels in (
                ("global", global_labels),
                ("within_cancer", within_labels),
            ):
                probability = fit_probability(
                    molecular_estimator(seed + iteration),
                    X_train,
                    labels,
                    X_test,
                    threads=threads,
                )
                rows.append(
                    {
                        "experiment": record["experiment"],
                        "dataset": record["name"],
                        "endpoint": record.get("endpoint"),
                        "permutation": permutation,
                        "iteration": iteration,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "roc_auc": float(roc_auc_score(y_test, probability)),
                        "pr_auc": float(average_precision_score(y_test, probability)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cohort and label-permutation stress tests.")
    parser.add_argument("--output-dir", default=str(RESULTS / "cancer/stress_tests"))
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--memory-gb", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)

    records = load_pooled_records()
    if args.smoke:
        records = [record for record in records if record["experiment"] == "extreme"]
        args.permutations = min(args.permutations, 3)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    held_out = held_out_tests(records, seed=args.seed, threads=args.threads)
    permutations = permutation_tests(
        records,
        iterations=args.permutations,
        seed=args.seed,
        threads=args.threads,
    )
    held_out.to_csv(output_dir / "cancer_cohort_held_out_metrics.csv", index=False)
    permutations.to_csv(output_dir / "cancer_label_permutation_metrics.csv", index=False)
    (output_dir / "cancer_stress_test_config.json").write_text(
        json.dumps(
            {
                "permutations": args.permutations,
                "seed": args.seed,
                "threads": args.threads,
                "memory_gb": args.memory_gb,
                "smoke": args.smoke,
                "foundation_models_executed": False,
                "feature_space": "existing preselected pooled feature space",
            },
            indent=2,
        )
    )
    print(f"held-out rows: {len(held_out)}; permutation rows: {len(permutations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
