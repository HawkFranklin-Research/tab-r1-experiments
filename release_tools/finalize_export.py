"""Prepare live configs and strip deployment metadata, not scientific results."""
from pathlib import Path
import json
import shutil
import sys

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]).resolve()
for name in ["batch_s2.json", "satya_recreation.json"]:
    cfg = json.loads((source / "Evaluate-TABPFN/configs" / name).read_text())
    cfg["output_root"] = "results/benchmark/refit_" + Path(name).stem
    for dataset in cfg["datasets"]:
        dataset["path"] = "01_benchmark_reproduction/datasets/" + Path(dataset["path"]).name
    target = root / "01_benchmark_reproduction/evaluator/configs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2) + "\n")

metadata = root / "results/cancer/cloud_models/EXECUTION_METADATA.json"
obj = json.loads(metadata.read_text())
def trim(value):
    if isinstance(value, dict):
        return {k: trim(v) for k, v in value.items()
                if not any(term in k.lower() for term in ["cost", "billing", "gcp_project", "networking", "instance_name"])}
    if isinstance(value, list):
        return [trim(v) for v in value]
    return value
metadata.write_text(json.dumps(trim(obj), indent=2) + "\n")
folder = root / "02_cancer_evaluation/cloud"
folder.mkdir(exist_ok=True)
shutil.copy2(source / "cloud/scripts/build_hf_fold_dataset.py", folder / "build_hf_fold_dataset.py")
print("Live benchmark configs made portable; deployment/billing metadata removed.")
