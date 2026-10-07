from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SHARED = Path("/home/prime/Documents/g3/cancer-os-exp/shared/scripts")
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from os_exp_common import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    ExportRecord,
    WINDOWS_DAYS,
    class_counts,
    dense_frame,
    discover_cancers,
    fixed_window_label,
    nonclinical_feature_positions,
    read_train_ready,
    top_variance_positions,
    write_manifest,
)


DEFAULT_OUTPUT_DIR = Path("/home/prime/Documents/g3/cancer-os-exp/exp01_per_cancer_fixed_window/datasets")


def _has_min_class_count(y: pd.Series, min_class_count: int) -> bool:
    return y.nunique() >= 2 and int(y.value_counts().min()) >= min_class_count


def export(input_root: Path, output_dir: Path, view: str, max_features: int, min_class_count: int) -> list[ExportRecord]:
    records: list[ExportRecord] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for cancer in discover_cancers(input_root, view):
        X, sample_index, feature_index = read_train_ready(input_root, cancer, view)
        if not {"OS_days", "OS_event"}.issubset(sample_index.columns):
            continue
        for endpoint, horizon_days in WINDOWS_DAYS.items():
            usable, y, excluded = fixed_window_label(sample_index, horizon_days)
            if usable.sum() < 50 or not _has_min_class_count(y, min_class_count):
                continue
            X_use = X[usable.to_numpy()]
            positions = top_variance_positions(X_use, nonclinical_feature_positions(feature_index), max_features)
            frame = dense_frame(X_use, feature_index, positions)
            frame["target"] = y.astype(str)

            name = f"{cancer}_{view}_{endpoint}_event_top{len(positions)}"
            dataset_path = output_dir / f"{name}.csv"
            metadata_path = output_dir / f"{name}_metadata.csv"
            frame.to_csv(dataset_path, index=False)

            metadata = sample_index.loc[usable].reset_index(drop=True).copy()
            metadata["cancer_type"] = cancer
            metadata["target"] = y.astype(str)
            metadata.to_csv(metadata_path, index=False)

            records.append(
                ExportRecord(
                    name=name,
                    path=str(dataset_path),
                    task="binary",
                    target="target",
                    source_cancers=[cancer],
                    view=view,
                    endpoint=endpoint,
                    horizon_days=horizon_days,
                    samples=len(frame),
                    features=len(positions),
                    class_counts=class_counts(y),
                    excluded_ambiguous=excluded,
                    metadata_path=str(metadata_path),
                    label_rule=(
                        f"1 = death observed on or before {horizon_days} days; "
                        f"0 = known survival beyond {horizon_days} days"
                    ),
                    note="Per-cancer fixed-window OS event classification. Clinical features are excluded.",
                )
            )

    write_manifest(
        output_dir,
        {
            "experiment": "exp01_per_cancer_fixed_window",
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "view": view,
            "max_features": max_features,
            "min_class_count": min_class_count,
            "feature_policy": "Top-variance non-clinical features selected independently per cancer/horizon dataset.",
            "label_policy": {
                "positive_class": "1 = death observed on or before horizon",
                "negative_class": "0 = known survival beyond horizon",
                "excluded": "patients censored before horizon",
                "horizons_days": WINDOWS_DAYS,
            },
        },
        records,
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-cancer fixed-window OS event datasets.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--view", default="core", choices=["core", "proteogenomic"])
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--min-class-count", type=int, default=5)
    args = parser.parse_args()

    records = export(
        input_root=Path(args.input_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        view=args.view,
        max_features=args.max_features,
        min_class_count=args.min_class_count,
    )
    print(json.dumps({"datasets_exported": len(records), "manifest": str(Path(args.output_dir) / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
