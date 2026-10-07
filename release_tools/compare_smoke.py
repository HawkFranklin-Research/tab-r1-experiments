"""Compare deterministic CSV values; reduced-draw uncertainty is not equivalent."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--rebuilt", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
report = {"passed": [], "differences": [], "skipped_columns": {}}
for folder in ["source_data/manuscript", "source_data/reusability", "tables"]:
    for old in sorted((root / "results" / folder).rglob("*.csv")):
        rel = old.relative_to(root / "results")
        new = args.rebuilt / rel
        if not new.exists():
            report["differences"].append({"file": str(rel), "issue": "missing"})
            continue
        a, b = pd.read_csv(old), pd.read_csv(new)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            report["differences"].append({"file": str(rel), "issue": "schema or shape"})
            continue
        skip = [c for c in a if any(x in c.lower() for x in
                ["ci_", "bootstrap", "p_did", "run", "path", "artifact", "file"])]
        report["skipped_columns"][str(rel)] = skip
        bad = []
        for c in a:
            if c in skip:
                continue
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                same = np.allclose(a[c], b[c], atol=1e-9, rtol=0, equal_nan=True)
            else:
                same = a[c].fillna("<NA>").astype(str).equals(b[c].fillna("<NA>").astype(str))
            if not same:
                bad.append(c)
        if bad:
            report["differences"].append({"file": str(rel), "columns": bad})
        else:
            report["passed"].append(str(rel))
print(json.dumps(report, indent=2))
