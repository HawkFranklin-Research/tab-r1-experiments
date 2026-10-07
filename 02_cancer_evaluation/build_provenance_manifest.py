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
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = REPO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path),
        "suffix": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        row["rows"] = len(frame)
        row["columns"] = json.dumps(frame.columns.tolist())
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a checksum-backed manifest for paper source data.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    paths = sorted(path for path in input_dir.rglob("*") if path.is_file())
    records = [describe(path) for path in paths]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps({"input_dir": str(input_dir), "files": records}, indent=2)
    )
    print(f"Recorded {len(records)} source artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
