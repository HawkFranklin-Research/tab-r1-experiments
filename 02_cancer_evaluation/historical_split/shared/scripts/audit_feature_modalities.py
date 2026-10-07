from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from os_exp_common import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    discover_cancers,
    modality_from_feature_id,
    nonclinical_feature_positions,
    read_train_ready,
    top_variance_positions,
)


DEFAULT_REPORTS_DIR = Path("/home/prime/Documents/g3/cancer-os-exp/shared/reports")


def _prefix_counts(feature_ids: pd.Series) -> dict[str, int]:
    modalities = feature_ids.astype(str).map(modality_from_feature_id)
    return {str(k): int(v) for k, v in modalities.value_counts().sort_index().to_dict().items()}


def audit(input_root: Path, reports_dir: Path, view: str, max_features: int) -> dict[str, str]:
    rows: list[dict[str, object]] = []
    prefix_rows: list[dict[str, object]] = []
    selected_prefix_rows: list[dict[str, object]] = []

    for cancer in discover_cancers(input_root, view):
        X, _sample_index, feature_index = read_train_ready(input_root, cancer, view)
        feature_ids = feature_index["feature_id"].astype(str)
        clinical_mask = feature_ids.str.startswith("clinical::")
        nonclinical_positions = nonclinical_feature_positions(feature_index)
        selected_positions = top_variance_positions(X, nonclinical_positions, max_features)
        selected_ids = feature_index.iloc[selected_positions]["feature_id"].astype(str)

        rows.append(
            {
                "cancer": cancer,
                "view": view,
                "total_features": int(len(feature_index)),
                "clinical_features": int(clinical_mask.sum()),
                "nonclinical_features": int((~clinical_mask).sum()),
                "selected_top_variance_features": int(len(selected_positions)),
                "feature_prefix_counts": json.dumps(_prefix_counts(feature_ids), sort_keys=True),
                "selected_prefix_counts": json.dumps(_prefix_counts(selected_ids), sort_keys=True),
            }
        )

        for modality, count in _prefix_counts(feature_ids).items():
            prefix_rows.append({"cancer": cancer, "view": view, "modality": modality, "count": count})
        for modality, count in _prefix_counts(selected_ids).items():
            selected_prefix_rows.append({"cancer": cancer, "view": view, "modality": modality, "count": count})

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "feature_modality_audit.csv"
    prefix_path = reports_dir / "feature_modality_prefix_counts.csv"
    selected_prefix_path = reports_dir / "feature_modality_selected_prefix_counts.csv"
    markdown_path = reports_dir / "feature_modality_audit.md"

    summary = pd.DataFrame(rows)
    prefix_df = pd.DataFrame(prefix_rows)
    selected_df = pd.DataFrame(selected_prefix_rows)
    summary.to_csv(summary_path, index=False)
    prefix_df.to_csv(prefix_path, index=False)
    selected_df.to_csv(selected_prefix_path, index=False)

    lines = [
        "# Feature Modality Audit",
        "",
        f"- Input root: `{input_root}`",
        f"- View: `{view}`",
        f"- Top-variance feature cap: `{max_features}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## All Feature Prefix Counts",
        "",
        prefix_df.to_markdown(index=False) if not prefix_df.empty else "_No features found._",
        "",
        "## Selected Top-Variance Prefix Counts",
        "",
        selected_df.to_markdown(index=False) if not selected_df.empty else "_No selected features found._",
        "",
    ]
    markdown_path.write_text("\n".join(lines))

    return {
        "summary_csv": str(summary_path),
        "prefix_counts_csv": str(prefix_path),
        "selected_prefix_counts_csv": str(selected_prefix_path),
        "markdown": str(markdown_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit feature modalities from train_ready feature_index.csv files.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--view", default="core", choices=["core", "proteogenomic"])
    parser.add_argument("--max-features", type=int, default=100)
    args = parser.parse_args()

    outputs = audit(
        input_root=Path(args.input_root).expanduser().resolve(),
        reports_dir=Path(args.reports_dir).expanduser().resolve(),
        view=args.view,
        max_features=args.max_features,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
