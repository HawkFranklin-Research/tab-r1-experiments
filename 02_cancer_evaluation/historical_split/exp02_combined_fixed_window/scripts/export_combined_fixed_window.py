from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = Path("/home/prime/Documents/g3/cancer-os-exp/shared/scripts")
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from os_exp_common import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    ExportRecord,
    WINDOWS_DAYS,
    cancer_label_counts,
    class_counts,
    discover_cancers,
    fixed_window_label,
    nonclinical_feature_positions,
    read_train_ready,
    safe_feature_name,
    top_variance_positions,
    write_manifest,
)


DEFAULT_OUTPUT_DIR = Path("/home/prime/Documents/g3/cancer-os-exp/exp02_combined_fixed_window/datasets")


def _has_min_class_count(y: pd.Series, min_class_count: int) -> bool:
    return y.nunique() >= 2 and int(y.value_counts().min()) >= min_class_count


def _selected_union(input_root: Path, cancers: list[str], view: str, endpoint: str, horizon_days: int, max_features: int, min_class_count: int) -> list[str]:
    per_cancer_top = max(1, max_features // max(1, len(cancers)))
    selected: list[str] = []
    for cancer in cancers:
        X, sample_index, feature_index = read_train_ready(input_root, cancer, view)
        usable, y, _excluded = fixed_window_label(sample_index, horizon_days)
        if usable.sum() < 50 or not _has_min_class_count(y, min_class_count):
            continue
        positions = top_variance_positions(X[usable.to_numpy()], nonclinical_feature_positions(feature_index), per_cancer_top)
        selected.extend(feature_index.iloc[positions]["feature_id"].astype(str).tolist())
    return sorted(set(selected))[:max_features]


def _dense_for_union(X, feature_index: pd.DataFrame, union_features: list[str]) -> pd.DataFrame:
    positions = feature_index["feature_index"].astype(int) if "feature_index" in feature_index.columns else pd.Series(range(len(feature_index)))
    feature_to_pos = dict(zip(feature_index["feature_id"].astype(str), positions))
    out = np.zeros((X.shape[0], len(union_features)), dtype=np.float32)
    present_cols: list[int] = []
    present_positions: list[int] = []
    for col_idx, feature_id in enumerate(union_features):
        pos = feature_to_pos.get(feature_id)
        if pos is not None:
            present_cols.append(col_idx)
            present_positions.append(pos)
    if present_positions:
        out[:, present_cols] = X[:, present_positions].toarray().astype(np.float32, copy=False)
    return pd.DataFrame(out, columns=[safe_feature_name(name) for name in union_features])


def export(input_root: Path, output_dir: Path, view: str, max_features: int, include_cancer_feature: bool, min_class_count: int) -> list[ExportRecord]:
    records: list[ExportRecord] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    cancers = discover_cancers(input_root, view)

    for endpoint, horizon_days in WINDOWS_DAYS.items():
        union_features = _selected_union(input_root, cancers, view, endpoint, horizon_days, max_features, min_class_count)
        if not union_features:
            continue
        frames: list[pd.DataFrame] = []
        metadata_frames: list[pd.DataFrame] = []

        for cancer in cancers:
            X, sample_index, feature_index = read_train_ready(input_root, cancer, view)
            usable, y, _excluded = fixed_window_label(sample_index, horizon_days)
            if usable.sum() < 50 or not _has_min_class_count(y, min_class_count):
                continue
            X_use = X[usable.to_numpy()]
            frame = _dense_for_union(X_use, feature_index, union_features)
            if include_cancer_feature:
                frame["cancer_type"] = cancer
            frame["target"] = y.astype(str)
            frames.append(frame)

            metadata = sample_index.loc[usable].reset_index(drop=True).copy()
            metadata["cancer_type"] = cancer
            metadata["target"] = y.astype(str)
            metadata_frames.append(metadata)

        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        metadata_combined = pd.concat(metadata_frames, ignore_index=True)
        cancer_suffix = "with_cancer_feature" if include_cancer_feature else "no_cancer_feature"
        name = f"ALL_{view}_{endpoint}_event_top{len(union_features)}_{cancer_suffix}"
        dataset_path = output_dir / f"{name}.csv"
        metadata_path = output_dir / f"{name}_metadata.csv"
        combined.to_csv(dataset_path, index=False)
        metadata_combined.to_csv(metadata_path, index=False)

        records.append(
            ExportRecord(
                name=name,
                path=str(dataset_path),
                task="binary",
                target="target",
                source_cancers=sorted(metadata_combined["cancer_type"].astype(str).unique().tolist()),
                view=view,
                endpoint=endpoint,
                horizon_days=horizon_days,
                samples=len(combined),
                features=len(combined.columns) - 1,
                class_counts=class_counts(combined["target"]),
                cancer_counts=class_counts(metadata_combined["cancer_type"]),
                cancer_label_counts=cancer_label_counts(metadata_combined["cancer_type"], combined["target"]),
                metadata_path=str(metadata_path),
                include_cancer_feature=include_cancer_feature,
                label_rule=(
                    f"1 = death observed on or before {horizon_days} days; "
                    f"0 = known survival beyond {horizon_days} days"
                ),
                note="Combined-cancer fixed-window OS event classification.",
            )
        )

    write_manifest(
        output_dir,
        {
            "experiment": "exp02_combined_fixed_window",
            "input_root": str(input_root),
            "output_dir": str(output_dir),
            "view": view,
            "max_features": max_features,
            "min_class_count": min_class_count,
            "include_cancer_feature": include_cancer_feature,
            "feature_policy": "Union of per-cancer top-variance non-clinical features, capped globally.",
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
    parser = argparse.ArgumentParser(description="Export combined-cancer fixed-window OS event datasets.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--view", default="core", choices=["core", "proteogenomic"])
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--include-cancer-feature", action="store_true")
    parser.add_argument("--min-class-count", type=int, default=5)
    args = parser.parse_args()

    records = export(
        input_root=Path(args.input_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        view=args.view,
        max_features=args.max_features,
        include_cancer_feature=args.include_cancer_feature,
        min_class_count=args.min_class_count,
    )
    print(json.dumps({"datasets_exported": len(records), "manifest": str(Path(args.output_dir) / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
