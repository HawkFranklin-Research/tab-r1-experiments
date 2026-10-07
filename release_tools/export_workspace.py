"""Curate a scientific code release without copying the workspace indiscriminately."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

DEST = Path(__file__).resolve().parents[1]
DENY = {"__pycache__", ".git", ".pytest_cache", "dist", "exploratory", "logs",
        "autogluon_models", "AutogluonModels", "catboost_info"}
ALLOWED = {".py", ".json", ".csv", ".npz", ".md", ".toml", ".yaml", ".yml"}
records = []


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    shutil.copy2(source, target)
    records.append({"source": source.name, "destination": str(target.relative_to(DEST)),
                    "sha256_original": hashlib.sha256(source.read_bytes()).hexdigest()})


def copy_tree(source, target, suffixes=ALLOWED):
    if not source.exists():
        return
    for p in sorted(source.rglob("*")):
        relative = p.relative_to(source)
        if not p.is_file() or any(x in DENY or x.endswith(".egg-info") or "smoke" in x.lower()
                                  for x in relative.parts):
            continue
        if p.suffix not in suffixes and p.name not in {"LICENSE", "Dockerfile"}:
            continue
        copy_file(p, target / relative)


def relocate(p):
    text = p.read_text()
    bootstrap = '''
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY
'''
    marker = "from __future__ import annotations"
    if marker not in text:
        raise ValueError(f"Missing future-import anchor in {p}")
    text = text.replace(marker, marker + "\n" + bootstrap, 1)
    replacements = {
        'Path(__file__).resolve().parents[2]': 'REPO',
        'ROOT / "package" / "src"': 'REPO / "ev_tabpfn/src"',
        'ROOT / "Evaluate-TABPFN"': 'REPO / "01_benchmark_reproduction/evaluator"',
        'ROOT / "cancer-os-exp"': 'REPO / "02_cancer_evaluation/historical_split"',
        'Path("/home/prime/Documents/g3/c-5/gpt/processed/train_ready")': 'TRAIN_READY',
        'ROOT / "paper" / "analysis" / "generated_folds"': 'FROZEN',
        'PAPER / "analysis/generated_folds"': 'FROZEN',
        'ANALYSIS / "generated_folds"': 'FROZEN',
        'PAPER / "tables/source_data/full_fold_models"': 'RESULTS / "cancer/local_models"',
        'PAPER / "tables/source_data/cloud_foundation_models/tabr1_results"': 'RESULTS / "cancer/cloud_models/tabr1_results"',
        'PAPER / "tables/source_data/cloud_foundation_models"': 'RESULTS / "cancer/cloud_models"',
        'PAPER / "tables/source_data/full_landscape"': 'RESULTS / "cancer/landscape"',
        'PAPER / "tables/source_data/full_stress"': 'RESULTS / "cancer/stress_tests"',
        'PAPER / "tables/source_data/full_cancer"': 'RESULTS / "cancer/shortcut_controls"',
        'PAPER / "figures/source_data/model_fold_metrics.csv"': 'RESULTS / "source_data/manuscript/model_fold_metrics.csv"',
        'PAPER / "figures/reusability/source_data"': 'RESULTS / "source_data/reusability"',
        'PAPER / "figures/source_data"': 'RESULTS / "source_data/manuscript"',
        'PAPER / "figures/manuscript"': 'RESULTS / "figures/manuscript"',
        'PAPER / "figures/reusability"': 'RESULTS / "figures/reusability"',
        'PAPER / "tables/generated"': 'RESULTS / "tables"',
        'REPO / "results-s2/runs/*/*/metrics/metrics_summary.csv"': 'RESULTS / "benchmark/run2_batch_s2/runs/*/*/metrics/metrics_summary.csv"',
        'REPO / "Evaluate-TABPFN/pfn3-test"': 'RESULTS / "benchmark/tabpfn_generations"',
        'REPO / "Evaluate-TABPFN/tabfm-test/reports/tabfm_satya_7_per_dataset_metrics.csv"': 'RESULTS / "benchmark/tabfm/reports/tabfm_satya_7_per_dataset_metrics.csv"',
        'REPO = ANALYSIS.parents[1]': '# REPO is supplied by tabr1_paths.',
        'SRC = OUT / "source_data"': 'SRC = RESULTS / "source_data/reusability"',
        'Path("paper/analysis/generated_folds")': 'FROZEN',
        'Path("/opt/workspace/tab-r1/paper/analysis/generated_folds")': 'FROZEN',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    p.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-preparation", type=Path, required=True)
    args = parser.parse_args()
    s = args.workspace.resolve()
    for name in ["src", "tests", "examples"]:
        copy_tree(s / "package" / name, DEST / "ev_tabpfn" / name)
    for name in ["pyproject.toml", "README.md", "LICENSE"]:
        copy_file(s / "package" / name, DEST / "ev_tabpfn" / name)
    copy_file(s / "LICENSE", DEST / "LICENSE")
    for name in ["preprocess_multiomics.py", "README.md"]:
        copy_file(args.data_preparation / name, DEST / "00_data_preparation" / name)
    bench = DEST / "01_benchmark_reproduction"
    copy_tree(s / "Evaluate-TABPFN/scripts", bench / "evaluator/scripts", {".py", ".md"})
    for p in (s / "Evaluate-TABPFN/tests").glob("*.py"):
        copy_file(p, bench / "evaluator/tests" / p.name)
    for name in ["run_pfn3_eval.py", "compare_generations.py"]:
        copy_file(s / "Evaluate-TABPFN/pfn3-test" / name, bench / "tabpfn_generations" / name)
    ce = DEST / "02_cancer_evaluation"
    for name in ["prepare_leakage_safe_folds", "run_leakage_safe_fold_models", "run_cloud_evaluation",
                 "run_cohort_stress_tests", "analyze_saved_cancer_results", "analyze_cancer_landscape",
                 "run_matched_baselines", "build_provenance_manifest", "resource_limits"]:
        copy_file(s / "paper/analysis" / f"{name}.py", ce / f"{name}.py")
        if name != "resource_limits":
            relocate(ce / f"{name}.py")
    copy_tree(s / "cancer-os-exp", ce / "historical_split", {".py", ".csv", ".json", ".npz"})
    af = DEST / "03_analysis_and_figures"
    for name in ["build_manuscript_assets.py", "within_cancer_pooling_contrast.py"]:
        copy_file(s / "paper/analysis" / name, af / name)
        relocate(af / name)
    for name in ["compute_reusability_statistics.py", "fold_aware_contrast.py", "build_reusability_figures.py"]:
        copy_file(s / "paper/analysis/reusability" / name, af / "reusability" / name)
        relocate(af / "reusability" / name)
    for old, new in [("full_fold_models", "local_models"), ("cloud_foundation_models", "cloud_models"),
                     ("full_cancer", "shortcut_controls"), ("full_stress", "stress_tests"),
                     ("full_landscape", "landscape")]:
        copy_tree(s / "paper/tables/source_data" / old, DEST / "results/cancer" / new,
                  {".csv", ".json", ".npz"})
    for name in ["results-satya", "results-s2"]:
        target = "run1_satya_recreation" if name == "results-satya" else "run2_batch_s2"
        copy_tree(s / name, DEST / "results/benchmark" / target, {".csv", ".json"})
    for name in ["outputs_generation_v1_cap", "outputs_generation_current_cap"]:
        copy_tree(s / "Evaluate-TABPFN/pfn3-test" / name,
                  DEST / "results/benchmark/tabpfn_generations" / name, {".csv", ".json"})
    copy_tree(s / "Evaluate-TABPFN/tabfm-test/reports", DEST / "results/benchmark/tabfm/reports", {".csv", ".json"})
    for old, new in [("paper/figures/source_data", "results/source_data/manuscript"),
                     ("paper/figures/reusability/source_data", "results/source_data/reusability"),
                     ("paper/tables/generated", "results/tables")]:
        copy_tree(s / old, DEST / new, {".csv", ".json"})
    for name in ["manuscript", "reusability"]:
        for p in (s / "paper/figures" / name).glob("*"):
            if p.is_file() and p.suffix in {".pdf", ".png", ".svg"}:
                copy_file(p, DEST / "results/figures" / name / p.name)
    copy_tree(s / "paper/analysis/generated_folds", DEST / "data/frozen_test_sets", {".csv", ".json"})
    for p in (s / "paper/tables/source_data").glob("full_provenance_manifest.*"):
        copy_file(p, DEST / "results/cancer/provenance" / p.name)
    env = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "scikit-survival",
                 "catboost", "xgboost", "lightgbm", "tabpfn", "autogluon.tabular", "torch", "huggingface-hub"]:
        try:
            env[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env[name] = None
    (DEST / "environment").mkdir(exist_ok=True)
    (DEST / "environment/observed_local_versions.json").write_text(json.dumps(env, indent=2) + "\n")
    (DEST / "export_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Copied {len(records)} files; source workspace unchanged.")


if __name__ == "__main__":
    main()
