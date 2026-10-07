from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tabr1_paths import REPO, FROZEN, TRAIN_READY

ROOT = REPO
DEFAULT_FOLD_ROOT = FROZEN
DEFAULT_OUTPUT_ROOT = REPO / "scratch/hf-fold-bundles"
DEFAULT_SOURCE_ROOT = TRAIN_READY
EXPECTED_FOLDS = 400
SPLITS = ("train", "validation", "test")
PATH_COLUMNS = {
    "selected_features_path",
    "train_path",
    "train_metadata_path",
    "validation_path",
    "validation_metadata_path",
    "test_path",
    "test_metadata_path",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def patient_ids(metadata: pd.DataFrame) -> set[str]:
    if "patient_id" in metadata.columns:
        values = metadata["patient_id"].fillna(metadata.get("sample_id"))
    elif "sample_id" in metadata.columns:
        values = metadata["sample_id"]
    else:
        raise ValueError("Metadata has neither patient_id nor sample_id")
    return set(values.astype(str))


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].astype(int).value_counts().to_dict()
    return {"class_0": int(counts.get(0, 0)), "class_1": int(counts.get(1, 0))}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def portable_fold_config(record: dict[str, Any], bundle_path: str) -> dict[str, Any]:
    config = {key: value for key, value in record.items() if key not in PATH_COLUMNS}
    config["bundle_path"] = bundle_path
    config["split_files"] = {
        split: {
            "features": f"{split}.parquet",
            "metadata": f"{split}_metadata.parquet",
        }
        for split in SPLITS
    }
    config["selected_features_file"] = "selected_features.parquet"
    config["feature_selection_statistics_file"] = "feature_selection_statistics.parquet"
    return config


def deterministic_archive(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "tar",
        "--zstd",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-cf",
        str(output_path),
        "-C",
        str(source_dir),
        ".",
    ]
    subprocess.run(command, check=True)


def build_bundle(record: dict[str, Any], output_root: Path) -> dict[str, Any]:
    scope = str(record["scope"])
    endpoint = str(record["endpoint"])
    cancer = str(record["cancer"])
    repeat = int(record["repeat"])
    fold = int(record["fold"])
    bundle_relative = (
        Path("folds")
        / scope
        / endpoint
        / cancer
        / f"repeat_{repeat:02d}"
        / f"fold_{fold:02d}.tar.zst"
    )
    bundle_path = output_root / bundle_relative

    frames: dict[str, pd.DataFrame] = {}
    metadata_frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        frame = pd.read_csv(record[f"{split}_path"])
        metadata_frame = pd.read_csv(record[f"{split}_metadata_path"])
        if len(frame) != len(metadata_frame):
            raise ValueError(f"{bundle_relative}: {split} data/metadata row mismatch")
        if "target" not in frame.columns:
            raise ValueError(f"{bundle_relative}: target missing from {split}")
        frames[split] = frame
        metadata_frames[split] = metadata_frame

    train_ids = patient_ids(metadata_frames["train"])
    validation_ids = patient_ids(metadata_frames["validation"])
    test_ids = patient_ids(metadata_frames["test"])
    overlaps = {
        "patient_overlap_train_validation": len(train_ids & validation_ids),
        "patient_overlap_train_test": len(train_ids & test_ids),
        "patient_overlap_validation_test": len(validation_ids & test_ids),
    }
    if any(overlaps.values()):
        raise ValueError(f"{bundle_relative}: patient overlap detected: {overlaps}")

    selected = pd.read_csv(record["selected_features_path"])
    feature_columns = [column for column in frames["train"].columns if column != "target"]
    if len(selected) != len(feature_columns):
        raise ValueError(f"{bundle_relative}: selected-feature count mismatch")
    selected = selected.copy()
    selected["column_name"] = feature_columns
    selected["train_variance"] = frames["train"][feature_columns].var(ddof=0).to_numpy(dtype=float)
    statistics = selected[["rank", "feature_id", "column_name", "modality", "train_variance"]].copy()

    with tempfile.TemporaryDirectory(prefix="tabr1-hf-fold-") as temporary:
        directory = Path(temporary)
        for split in SPLITS:
            frames[split].to_parquet(directory / f"{split}.parquet", index=False, compression="zstd")
            metadata_frames[split].to_parquet(
                directory / f"{split}_metadata.parquet", index=False, compression="zstd"
            )
        selected.drop(columns=["train_variance"]).to_parquet(
            directory / "selected_features.parquet", index=False, compression="zstd"
        )
        statistics.to_parquet(
            directory / "feature_selection_statistics.parquet", index=False, compression="zstd"
        )
        write_json(directory / "fold_config.json", portable_fold_config(record, bundle_relative.as_posix()))
        deterministic_archive(directory, bundle_path)

    split_counts = {split: class_counts(frames[split]) for split in SPLITS}
    modalities = selected["modality"].astype(str).value_counts().sort_index().to_dict()
    return {
        **{key: record[key] for key in record if key not in PATH_COLUMNS},
        **overlaps,
        "class_0_train": split_counts["train"]["class_0"],
        "class_0_validation": split_counts["validation"]["class_0"],
        "class_0_test": split_counts["test"]["class_0"],
        "bundle_path": bundle_relative.as_posix(),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256(bundle_path),
        "selected_modality_counts": json.dumps(modalities, sort_keys=True),
    }


def source_provenance(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cancer in ("BRCA", "ESCA", "HNSCC", "LSCC", "LUAD"):
        for name in ("X.npz", "sample_index.csv", "feature_index.csv"):
            path = source_root / cancer / "core" / name
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "path": f"train_ready/{cancer}/core/{name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return rows


def write_dataset_files(
    output_root: Path,
    records: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    frame = pd.DataFrame(records).sort_values(["scope", "endpoint", "cancer", "repeat", "fold"])
    frame.to_csv(output_root / "manifest.csv", index=False)
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": "1.0.0",
            "generated_at": generated_at,
            "fold_count": len(records),
            "records": frame.to_dict(orient="records"),
        },
    )
    write_json(
        output_root / "label_contract.json",
        {
            "task_type": "binary classification",
            "os_3yr": {
                "horizon_days": 1095,
                "class_1": "OS_event = 1 and OS_days <= 1095",
                "class_0": "OS_days > 1095, including death after the horizon or known follow-up beyond it",
                "excluded": "censored before 1095 days or missing OS fields",
            },
            "os_5yr": {
                "horizon_days": 1825,
                "class_1": "OS_event = 1 and OS_days <= 1825",
                "class_0": "OS_days > 1825, including death after the horizon or known follow-up beyond it",
                "excluded": "censored before 1825 days or missing OS fields",
            },
            "extreme_os": {
                "class_1": "OS_event = 1 and OS_days < 1095",
                "class_0": "OS_event = 0 and OS_days >= 1825",
                "excluded": "all intermediate, late-death, early-censored, or missing outcomes",
            },
        },
    )
    write_json(
        output_root / "provenance.json",
        {
            "generated_at": generated_at,
            "repository": "HawkFranklin-Research/tab-r1",
            "git_commit": git_value("rev-parse", "HEAD"),
            "fold_exporter": "paper/analysis/prepare_leakage_safe_folds.py",
            "hub_packager": "cloud/scripts/build_hf_fold_dataset.py",
            "source_description": "Processed TCGA and CPTAC/LinkedOmics core-view multiomics matrices",
            "clinical_features_excluded": True,
            "feature_harmonization": "Exact common non-clinical feature identifiers for pooled cohorts",
            "feature_selection": "Top 100 variance-ranked features fitted independently on training rows",
            "source_files": source_files,
        },
    )
    write_json(
        output_root / "software_environment.json",
        {
            "generated_at": generated_at,
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "packages": {
                name: package_version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "pyarrow",
                    "catboost",
                    "xgboost",
                    "lightgbm",
                    "huggingface-hub",
                )
            },
        },
    )
    write_json(
        output_root / "model_execution_matrix.json",
        {
            "dataset_revision_policy": "Pin the Hugging Face dataset commit SHA for every run",
            "common_contract": {
                "fold_count": EXPECTED_FOLDS,
                "seed": 42,
                "threshold_selection": "Validation-set balanced accuracy only",
                "maximum_features": 100,
            },
            "local_completed": [
                "logistic_regression",
                "random_forest",
                "catboost",
                "xgboost",
                "lightgbm",
            ],
            "cloud_pending": [
                {"model": "autogluon", "hardware": "cpu-upgrade"},
                {"model": "tabpfn_v2", "hardware": "l4x1", "context_cap": 1024},
                {"model": "tabpfn_v2_5", "hardware": "l4x1", "context_cap": 1024},
                {"model": "tabpfn_v2_6", "hardware": "l4x1", "context_cap": 1024},
                {"model": "tabpfn_v3", "hardware": "l4x1", "context_cap": 1024},
                {"model": "tabfm_default", "hardware": "l4x1", "context_cap": 1024},
            ],
        },
    )

    readme = """---
license: other
task_categories:
- tabular-classification
tags:
- cancer
- multiomics
- overall-survival
- tabular-foundation-models
- benchmark
pretty_name: TABR1 Cancer OS Leakage-Safe Folds v1
---

# TABR1 Cancer OS Leakage-Safe Folds v1

This private research dataset contains 400 frozen, patient-grouped evaluation folds for fixed-window and extreme-contrast overall-survival classification across BRCA, ESCA, HNSCC, LSCC, and LUAD cohorts.

## Scientific contract

- Five repeats and five outer folds per eligible task/cohort combination.
- Approximately 64% training, 16% validation, and 20% testing.
- Patient identifiers are grouped so a patient cannot cross splits within a fold.
- Clinical features are excluded.
- Feature selection is variance ranking fitted on training rows only.
- Pooled tasks use the exact common non-clinical feature universe across cancers.
- Decision thresholds must be selected using validation data only.

The endpoints are defined in `label_contract.json`. This is binary endpoint classification, not continuous censored-survival modeling.

## Contents

Every `folds/**/*.tar.zst` bundle contains:

```text
train.parquet
validation.parquet
test.parquet
train_metadata.parquet
validation_metadata.parquet
test_metadata.parquet
selected_features.parquet
feature_selection_statistics.parquet
fold_config.json
```

`manifest.csv` and `manifest.json` provide sample counts, class counts, patient-overlap audits, feature/modality summaries, bundle checksums, and portable bundle paths.

## Loading one fold

```bash
python examples/load_one_fold.py \
  --dataset-root . \
  --scope pooled \
  --endpoint os_3yr \
  --cancer ALL \
  --repeat 0 \
  --fold 0
```

## Provenance and use restrictions

These are derived research artifacts from locally processed TCGA and CPTAC/LinkedOmics data. The repository is private while provenance, source terms, and redistribution requirements are reviewed. Users remain responsible for complying with the original data-source terms. Do not use these exploratory endpoints for clinical decision-making.
"""
    (output_root / "README.md").write_text(readme)


def write_example(output_root: Path) -> None:
    example = '''from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one TABR1 fold bundle.")
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--scope", required=True, choices=["per_cancer", "pooled"])
    parser.add_argument("--endpoint", required=True, choices=["os_3yr", "os_5yr", "extreme_os"])
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int)
    args = parser.parse_args()

    bundle = (
        args.dataset_root
        / "folds"
        / args.scope
        / args.endpoint
        / args.cancer
        / f"repeat_{args.repeat:02d}"
        / f"fold_{args.fold:02d}.tar.zst"
    )
    if not bundle.exists():
        raise FileNotFoundError(bundle)

    with tempfile.TemporaryDirectory(prefix="tabr1-fold-") as temporary:
        subprocess.run(["tar", "--zstd", "-xf", str(bundle), "-C", temporary], check=True)
        root = Path(temporary)
        train = pd.read_parquet(root / "train.parquet")
        validation = pd.read_parquet(root / "validation.parquet")
        test = pd.read_parquet(root / "test.parquet")
        print({"train": train.shape, "validation": validation.shape, "test": test.shape})


if __name__ == "__main__":
    main()
'''
    examples = output_root / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    (examples / "load_one_fold.py").write_text(example)


def write_checksums(output_root: Path) -> None:
    checksum_path = output_root / "checksums.sha256"
    files = sorted(path for path in output_root.rglob("*") if path.is_file() and path != checksum_path)
    with checksum_path.open("w") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}\n")


def verify(output_root: Path) -> None:
    manifest = pd.read_csv(output_root / "manifest.csv")
    if len(manifest) != EXPECTED_FOLDS:
        raise ValueError(f"Expected {EXPECTED_FOLDS} folds, found {len(manifest)}")
    if manifest["bundle_path"].nunique() != EXPECTED_FOLDS:
        raise ValueError("Bundle paths are not unique")
    if manifest[[
        "patient_overlap_train_validation",
        "patient_overlap_train_test",
        "patient_overlap_validation_test",
    ]].to_numpy().max() != 0:
        raise ValueError("Patient overlap found in manifest")
    for row in manifest.itertuples(index=False):
        bundle = output_root / row.bundle_path
        if not bundle.exists() or sha256(bundle) != row.bundle_sha256:
            raise ValueError(f"Bundle verification failed: {row.bundle_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the private TABR1 Hugging Face fold dataset.")
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fold_root = args.fold_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    manifest_path = fold_root / "fold_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_manifest = pd.read_csv(manifest_path)
    if len(source_manifest) != EXPECTED_FOLDS:
        raise ValueError(f"Expected {EXPECTED_FOLDS} source folds, found {len(source_manifest)}")
    records: list[dict[str, Any]] = []
    for index, row in source_manifest.iterrows():
        records.append(build_bundle(row.to_dict(), output_root))
        if (index + 1) % 25 == 0:
            print(f"Packaged {index + 1}/{EXPECTED_FOLDS} folds", flush=True)

    write_dataset_files(output_root, records, source_provenance(source_root))
    write_example(output_root)
    write_checksums(output_root)
    verify(output_root)
    print(f"Built and verified {len(records)} fold bundles at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
