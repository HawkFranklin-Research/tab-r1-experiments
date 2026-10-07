#!/usr/bin/env python3
"""Run frozen TABR1 folds from the private Hugging Face dataset in the cloud."""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
PACKAGE_SRC = REPO / "ev_tabpfn/src"
for path in (SCRIPT_DIR, PACKAGE_SRC, ROOT_DIR / "tabfm"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from resource_limits import configure_process_limits
from run_leakage_safe_fold_models import AVAILABLE_MODELS, FOUNDATION_MODELS, execute_model, parse_list


DEFAULT_HF_REPO = "HawkFranklin-Research/TABR1-Cancer-OS-LeakageSafe-Folds"
DEFAULT_HF_REVISION = "0b26eb0bf0eae7558a6a659aea7829209b31854e"
EXPECTED_FOLDS = 400
REQUIRED_MEMBERS = {
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "train_metadata.parquet",
    "validation_metadata.parquet",
    "test_metadata.parquet",
    "selected_features.parquet",
    "feature_selection_statistics.parquet",
    "fold_config.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for the private dataset. Inject it with Secret Manager "
            "or export it in the VM shell; never pass it as a command-line argument."
        )
    return token


def verify_snapshot(dataset_root: Path) -> None:
    manifest_path = dataset_root / "manifest.csv"
    checksum_path = dataset_root / "checksums.sha256"
    if not manifest_path.exists() or not checksum_path.exists():
        raise FileNotFoundError("Downloaded dataset is missing manifest.csv or checksums.sha256")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != EXPECTED_FOLDS or manifest["bundle_path"].nunique() != EXPECTED_FOLDS:
        raise ValueError(f"Expected {EXPECTED_FOLDS} unique folds, found {len(manifest)}")
    failures: list[str] = []
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = dataset_root / relative
        if not path.exists() or sha256(path) != expected:
            failures.append(relative)
    if failures:
        raise ValueError(f"Dataset checksum failures: {failures[:10]}")


def download_hf_dataset(repo_id: str, revision: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        token=require_hf_token(),
        max_workers=8,
    )
    verify_snapshot(local_dir)
    return local_dir / "manifest.csv"


def materialize_fold(row: pd.Series, dataset_root: Path, cache_root: Path) -> pd.Series:
    archive = dataset_root / str(row["bundle_path"])
    expected_sha = str(row["bundle_sha256"])
    if not archive.exists() or sha256(archive) != expected_sha:
        raise ValueError(f"Fold archive failed verification: {archive}")
    fold_dir = (
        cache_root
        / str(row["scope"])
        / str(row["endpoint"])
        / str(row["cancer"])
        / f"repeat_{int(row['repeat']):02d}_fold_{int(row['fold']):02d}"
    )
    marker = fold_dir / ".bundle_sha256"
    if not marker.exists() or marker.read_text().strip() != expected_sha:
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        fold_dir.mkdir(parents=True)
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(fold_dir)], check=True)
        members = {path.name for path in fold_dir.iterdir() if path.is_file()}
        if not REQUIRED_MEMBERS.issubset(members):
            raise ValueError(f"Incomplete fold bundle {archive}: {sorted(members)}")
        marker.write_text(expected_sha + "\n")
    materialized = row.copy()
    for split in ("train", "validation", "test"):
        materialized[f"{split}_path"] = str(fold_dir / f"{split}.parquet")
        materialized[f"{split}_metadata_path"] = str(fold_dir / f"{split}_metadata.parquet")
    materialized["selected_features_path"] = str(fold_dir / "selected_features.parquet")
    return materialized


def upload_results(output_root: Path, repo_id: str, task_index: int, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_root,
        path_in_repo=f"tasks/task_{task_index:03d}",
        commit_message=f"Upload TABR1 cloud results for task {task_index}",
        ignore_patterns=["fold_cache/**", "hf_dataset/**", "**/autogluon_model/**"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--dataset-dir", default="/tmp/tabr1_hf_dataset")
    parser.add_argument("--output-root", default="/tmp/tabr1_outputs")
    parser.add_argument("--models", default=os.environ.get("TABR1_MODELS", "tabpfn_v3,tabfm_default"))
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0")))
    parser.add_argument("--task-count", type=int, default=int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1")))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-gb", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tabfm-backend", choices=["pytorch", "jax"], default="pytorch")
    parser.add_argument("--autogluon-time-limit", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--hf-results-repo", default=os.environ.get("HF_RESULTS_REPO"))
    args = parser.parse_args()

    configure_process_limits(threads=args.threads, memory_gb=args.memory_gb)
    dataset_root = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = download_hf_dataset(args.hf_repo, args.hf_revision, dataset_root)
    if args.download_only:
        print(f"Verified {EXPECTED_FOLDS} downloaded folds at revision {args.hf_revision}")
        return 0

    models = parse_list(args.models)
    unknown = sorted(set(models) - set(AVAILABLE_MODELS))
    if unknown:
        parser.error(f"Unknown models: {', '.join(unknown)}")
    manifest = pd.read_csv(manifest_path).iloc[args.start_fold : args.end_fold]
    if args.max_folds is not None:
        manifest = manifest.head(args.max_folds)
    if not 0 <= args.task_index < args.task_count:
        parser.error("task-index must satisfy 0 <= task-index < task-count")
    manifest = manifest.iloc[args.task_index :: args.task_count]
    if len(manifest) > 5 and set(models) & FOUNDATION_MODELS and not args.confirm_full_run:
        parser.error("Foundation-model execution across more than five folds requires --confirm-full-run")

    cache_root = output_root / "fold_cache"
    rows: list[dict[str, Any]] = []
    started = time.time()
    total = len(manifest) * len(models)
    run_index = 0
    for _, packed_fold in manifest.iterrows():
        fold = materialize_fold(packed_fold, dataset_root, cache_root)
        for model_name in models:
            run_index += 1
            run_dir = (
                output_root
                / str(fold["scope"])
                / str(fold["endpoint"])
                / str(fold["cancer"])
                / f"repeat_{int(fold['repeat']):02d}_fold_{int(fold['fold']):02d}"
                / model_name
            )
            result_file = run_dir / "result.json"
            if args.resume and result_file.exists():
                cached = json.loads(result_file.read_text())
                if cached.get("status") == "success":
                    rows.append(cached)
                    print(f"[{run_index}/{total}] cached {model_name}", flush=True)
                    continue
            print(f"[{run_index}/{total}] running {model_name}", flush=True)
            rows.append(
                execute_model(
                    fold,
                    model_name,
                    output_root=output_root,
                    seed=args.seed,
                    threads=args.threads,
                    tabfm_backend=args.tabfm_backend,
                    autogluon_time_limit=args.autogluon_time_limit,
                    device=args.device,
                )
            )

    results = pd.DataFrame(rows)
    results.to_csv(output_root / "all_fold_model_metrics.csv", index=False)
    (output_root / "run_config.json").write_text(
        json.dumps(
            {
                "hf_repo": args.hf_repo,
                "hf_revision": args.hf_revision,
                "models": models,
                "task_index": args.task_index,
                "task_count": args.task_count,
                "folds": len(manifest),
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    if args.hf_results_repo:
        upload_results(output_root, args.hf_results_repo, args.task_index, require_hf_token())
    failures = int((results.get("status", pd.Series(dtype=str)) != "success").sum())
    print(f"Completed {len(results)} runs with {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
