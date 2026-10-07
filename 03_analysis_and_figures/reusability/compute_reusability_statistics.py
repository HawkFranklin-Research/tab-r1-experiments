"""Statistics for the reusability-report version of the manuscript.

Reads saved metrics and predictions only; no model is fitted. Writes tidy CSVs to
paper/figures/reusability/source_data/ for build_reusability_figures.py and the text.

Outputs
- reproduction_auc.csv            small-dataset AUC per dataset and model (historical runs)
- within_cancer_ranks.csv         per-task mean AUC and rank for the 13 cancer-specific tasks
- within_cancer_friedman.csv      Friedman test and Nemenyi critical difference
- paired_delta_vs_rf.csv          fold-matched AUC difference from random forest
- legacy_patient_averaged/*.csv   SUPERSEDED patient-averaged pooling estimator (see fold_aware_contrast.py for the current one)
- pooling_did.csv                 foundation-model minus tree-ensemble delta (difference-in-differences)
- pooling_share_harmed.csv        share of cancers in which pooled training lowered within-cancer AUC
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "12")

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from within_cancer_pooling_contrast import auc, load_patient_probabilities  # noqa: E402

# REPO is supplied by tabr1_paths.
PAPER = REPO / "paper"
METRICS_PATH = RESULTS / "source_data/manuscript/model_fold_metrics.csv"
OUT = RESULTS / "source_data/reusability"

FOUNDATION = ["tabfm_default", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"]
TREES = ["random_forest", "catboost", "xgboost", "lightgbm"]
N_BOOT = 2000
SEED = 20261002
# Studentized range q_0.05 / sqrt(2) for k = 11 classifiers (Demsar 2006, Table 5).
NEMENYI_Q05_K11 = 3.219


def reproduction_table() -> pd.DataFrame:
    rows = []
    # Baseline comparison (results-s2): one stratified 85/15 split per dataset; latest run per dataset.
    for path in sorted(glob.glob(str(RESULTS / "benchmark/run2_batch_s2/runs/*/*/metrics/metrics_summary.csv"))):
        dataset = path.split("/runs/")[1].split("/")[0]
        frame = pd.read_csv(path)
        frame = frame[frame["status"] == "success"]
        for r in frame.itertuples():
            rows.append({"dataset": dataset, "model": r.model_name, "roc_auc": r.roc_auc, "study": "baselines", "run": path})
    # Generation comparison (Evaluate-TABPFN/pfn3-test, training capped at 1,024 rows).
    for summary in ["outputs_generation_v1_cap", "outputs_generation_current_cap"]:
        batch = json.load(open(RESULTS / "benchmark/tabpfn_generations" / summary / "batch_summary.json"))
        for run in batch["runs"]:
            for r in run["metrics"]["rows"]:
                if r.get("status") == "success":
                    rows.append({"dataset": run["dataset_name"], "model": f"tabpfn_{r['version']}", "roc_auc": r["roc_auc"], "study": "generations", "run": run["output_dir"]})
    tabfm = pd.read_csv(RESULTS / "benchmark/tabfm/reports/tabfm_satya_7_per_dataset_metrics.csv")
    for r in tabfm[tabfm["status"] == "success"].itertuples():
        rows.append({"dataset": r.dataset, "model": "tabfm", "roc_auc": r.roc_auc, "study": "generations", "run": r.run_dir})
    table = pd.DataFrame(rows)
    table["dataset"] = table["dataset"].str.replace("-", "_").str.replace("_dataset", "")
    table = table.sort_values("run").groupby(["study", "dataset", "model"], as_index=False).last()
    return table


def within_cancer_statistics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per = metrics[metrics["scope"] == "per_cancer"]
    task = per.groupby(["endpoint", "cancer", "model_name"])["roc_auc"].mean().unstack()
    ranks = task.apply(lambda row: pd.Series(rankdata(-row.to_numpy()), index=row.index), axis=1)
    long = task.stack().rename("roc_auc").reset_index().merge(
        ranks.stack().rename("rank").reset_index(), on=["endpoint", "cancer", "model_name"])
    stat, p = friedmanchisquare(*[task[c].to_numpy() for c in task.columns])
    k, n = task.shape[1], task.shape[0]
    cd = NEMENYI_Q05_K11 * np.sqrt(k * (k + 1) / (6 * n))
    friedman = pd.DataFrame([{"n_tasks": n, "n_models": k, "chi2": stat, "p_value": p, "nemenyi_cd_alpha05": cd}])
    rf = per[per["model_name"] == "random_forest"][["endpoint", "cancer", "repeat", "fold", "roc_auc"]]
    delta = per.merge(rf, on=["endpoint", "cancer", "repeat", "fold"], suffixes=("", "_rf"))
    delta["delta_vs_rf"] = delta["roc_auc"] - delta["roc_auc_rf"]
    delta = delta[delta["model_name"] != "random_forest"][["endpoint", "cancer", "repeat", "fold", "model_name", "delta_vs_rf"]]
    return long, friedman, delta


def pooling_bootstrap(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probs = load_patient_probabilities(metrics)
    wide = probs.pivot_table(
        index=["endpoint", "model_name", "cancer_type", "patient_id", "y_true"], columns="scope", values="probability",
    ).dropna(subset=["per_cancer", "pooled"]).reset_index()
    rng = np.random.default_rng(SEED)
    summary_rows, did_rows, harm_rows = [], [], []
    for endpoint, ep in wide.groupby("endpoint"):
        models = sorted(ep["model_name"].unique())
        cancers = sorted(ep["cancer_type"].unique())
        draws = {m: np.zeros((N_BOOT, len(cancers))) for m in models}
        point = {m: np.zeros(len(cancers)) for m in models}
        for j, cancer in enumerate(cancers):
            # Shared patient indices so every model is resampled on the same patients.
            base = ep[(ep["cancer_type"] == cancer)].pivot_table(
                index=["patient_id", "y_true"], columns="model_name", values=["per_cancer", "pooled"]).dropna()
            y = base.index.get_level_values("y_true").to_numpy(int)
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            idx = [np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)]) for _ in range(N_BOOT)]
            for m in models:
                pc, pp = base[("per_cancer", m)].to_numpy(), base[("pooled", m)].to_numpy()
                point[m][j] = auc(y, pp) - auc(y, pc)
                draws[m][:, j] = [auc(y[i], pp[i]) - auc(y[i], pc[i]) for i in idx]
        for m in models:
            mean_draw = draws[m].mean(axis=1)
            summary_rows.append({"endpoint": endpoint, "model_name": m, "n_cancers": len(cancers),
                                 "mean_delta": point[m].mean(), "ci_low": np.percentile(mean_draw, 2.5),
                                 "ci_high": np.percentile(mean_draw, 97.5)})
            harm_rows.append({"endpoint": endpoint, "model_name": m, "n_cancers": len(cancers),
                              "n_harmed": int((point[m] < 0).sum()), "share_harmed": float((point[m] < 0).mean())})
        fm = [m for m in FOUNDATION if m in draws]
        tr = [m for m in TREES if m in draws]
        fm_draw = np.mean([draws[m].mean(axis=1) for m in fm], axis=0)
        tr_draw = np.mean([draws[m].mean(axis=1) for m in tr], axis=0)
        did = fm_draw - tr_draw
        did_rows.append({"endpoint": endpoint, "foundation_models": ";".join(fm), "tree_models": ";".join(tr),
                         "foundation_mean_delta": np.mean([point[m].mean() for m in fm]),
                         "tree_mean_delta": np.mean([point[m].mean() for m in tr]),
                         "did": np.mean([point[m].mean() for m in fm]) - np.mean([point[m].mean() for m in tr]),
                         "did_ci_low": np.percentile(did, 2.5), "did_ci_high": np.percentile(did, 97.5),
                         "p_did_ge_0": float((did >= 0).mean())})
    return pd.DataFrame(summary_rows), pd.DataFrame(did_rows), pd.DataFrame(harm_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    reproduction_table().to_csv(OUT / "reproduction_auc.csv", index=False)
    ranks, friedman, delta = within_cancer_statistics(metrics)
    ranks.to_csv(OUT / "within_cancer_ranks.csv", index=False)
    friedman.to_csv(OUT / "within_cancer_friedman.csv", index=False)
    delta.to_csv(OUT / "paired_delta_vs_rf.csv", index=False)
    # Legacy estimator (patient-averaged predictions pooled across folds). Superseded by fold_aware_contrast.py;
    # kept only for the estimator-sensitivity panel (Extended Data Fig. 3E). Does not feed Figure 5.
    summary, did, harm = pooling_bootstrap(metrics)
    legacy = OUT / "legacy_patient_averaged"
    legacy.mkdir(exist_ok=True)
    summary.to_csv(legacy / "pooling_delta_draws_summary.csv", index=False)
    did.to_csv(legacy / "pooling_did.csv", index=False)
    harm.to_csv(legacy / "pooling_share_harmed.csv", index=False)
    with pd.option_context("display.width", 220):
        print(friedman.round(4).to_string(index=False))
        print(ranks.groupby("model_name")["rank"].mean().sort_values().round(2).to_string())
        print(did.drop(columns=["foundation_models", "tree_models"]).round(3).to_string(index=False))
        print(harm.pivot(index="model_name", columns="endpoint", values="n_harmed").to_string())


if __name__ == "__main__":
    main()
