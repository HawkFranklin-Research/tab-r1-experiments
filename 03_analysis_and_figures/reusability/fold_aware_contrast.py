"""Fold-aware within-cancer pooling contrast (ROC AUC and Harrell C-index).

Replaces the patient-averaged estimator of ``within_cancer_pooling_contrast.py`` for the
reusability report. That estimator averaged each patient's predictions over repeats and scored
one AUC per cancer over all patients, which pools predictions made by different models in
different folds; between-fold shifts in score scale then enter the AUC. Here every test set is
scored on its own, exactly as in Table 2, and the pooled-minus-cancer-specific difference is the
difference of fold-level means.

Uncertainty: a patient-cluster bootstrap. Within each endpoint and cancer, patients are resampled
with replacement separately for each outcome class; the same resampled patients are used in every
model, in both scopes, in every repeat and fold (weights, not copies), so differences between
scopes and between model families stay paired. Reads saved predictions only; fits no model.

Metrics
- ``auc``: ROC AUC within the test set (ties count one half).
- ``cidx``: Harrell C-index among the eligible patients of the test set, using each patient's real
  overall-survival time and event and the saved horizon-classifier probability as the risk score.
  Patients censored before the horizon were never in the folds, so this is not a
  censoring-adjusted C-index over the whole cohort.

Outputs (paper/figures/reusability/source_data/)
- foldaware_pooling_summary.csv    per metric/endpoint/model: change after pooling with 95% CI
- foldaware_pooling_per_cancer.csv per metric/endpoint/model/cancer: change with 95% CI
- foldaware_pooling_did.csv        foundation-model mean change minus tree-ensemble mean change
- foldaware_pooling_harmed.csv     number of cancers in which pooling lowered the metric
- cindex_eligible_summary.csv      fold-level AUC and Harrell C by model, endpoint and scoring scope
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

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))
from within_cancer_pooling_contrast import prediction_path  # noqa: E402

PAPER = ANALYSIS.parent
FOLD_ROOT = FROZEN
METRICS_PATH = RESULTS / "source_data/manuscript/model_fold_metrics.csv"
OUT = RESULTS / "source_data/reusability"

FOUNDATION = ["tabfm_default", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"]
TREES = ["random_forest", "catboost", "xgboost", "lightgbm"]
N_BOOT = 2000
SEED = 20261003


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def load_cells(metrics: pd.DataFrame) -> list[dict]:
    """One record per saved test set: patient ids, cancer, label, risk, survival time and event."""
    cells = []
    for _, row in metrics.iterrows():
        path = prediction_path(row)
        if not path.exists():
            continue
        fold_dir = FOLD_ROOT / row["scope"] / row["endpoint"] / row["cancer"] / f"repeat_{int(row['repeat']):02d}_fold_{int(row['fold']):02d}"
        pred = pd.read_csv(path, usecols=["sample_id", "patient_id", "cancer_type", "y_true", "probability"])
        meta = pd.read_csv(fold_dir / "test_metadata.csv", usecols=["sample_id", "OS_days", "OS_event"])
        frame = pred.merge(meta, on="sample_id", how="left", validate="one_to_one")
        if frame[["OS_days", "OS_event"]].isna().any().any():
            raise ValueError(f"missing survival data in {fold_dir}")
        cells.append({
            "endpoint": row["endpoint"], "model": row["model_name"], "scope": row["scope"],
            "repeat": int(row["repeat"]), "fold": int(row["fold"]),
            "patient": frame["patient_id"].to_numpy(), "cancer": frame["cancer_type"].to_numpy(),
            "y": frame["y_true"].to_numpy(int), "risk": frame["probability"].to_numpy(float),
            "time": frame["OS_days"].to_numpy(float), "event": frame["OS_event"].to_numpy(float) > 0.5,
        })
    return cells


def build_universes(cells: list[dict]) -> dict:
    """Per (endpoint, cancer): patient list, class labels, and bootstrap weights (D+1 draws; row 0 = ones)."""
    seen: dict[tuple, dict[str, int]] = defaultdict(dict)
    for c in cells:
        for pid, canc, y in zip(c["patient"], c["cancer"], c["y"]):
            key = (c["endpoint"], canc)
            if pid in seen[key] and seen[key][pid] != y:
                raise ValueError(f"inconsistent label for {pid} in {key}")
            seen[key][pid] = int(y)
    rng = np.random.default_rng(SEED)
    universes = {}
    for key in sorted(seen):
        patients = sorted(seen[key])
        y = np.array([seen[key][p] for p in patients])
        weights = np.ones((N_BOOT + 1, len(patients)), dtype=np.float64)
        groups = [np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
        for d in range(1, N_BOOT + 1):
            counts = np.zeros(len(patients))
            for g in groups:
                counts += np.bincount(rng.choice(g, size=g.size), minlength=len(patients))
            weights[d] = counts
        universes[key] = {"index": {p: i for i, p in enumerate(patients)}, "W": weights}
    return universes


# ---------------------------------------------------------------------------------------------
# Weighted metrics over all draws at once
# ---------------------------------------------------------------------------------------------
def weighted_metrics(W: np.ndarray, idx: np.ndarray, y: np.ndarray, risk: np.ndarray, time_: np.ndarray, event: np.ndarray):
    """AUC and Harrell C of one test-set subset under every bootstrap weight vector (row 0 = observed)."""
    w = W[:, idx]
    d = W.shape[0]
    pos, neg = y == 1, y == 0
    if pos.sum() and neg.sum():
        sp, sn = risk[pos], risk[neg]
        pair = (sp[:, None] > sn[None, :]).astype(float) + 0.5 * (sp[:, None] == sn[None, :])
        wp, wn = w[:, pos], w[:, neg]
        den = wp.sum(1) * wn.sum(1)
        auc = np.where(den > 0, ((wp @ pair) * wn).sum(1) / np.where(den > 0, den, 1), np.nan)
    else:
        auc = np.full(d, np.nan)
    comparable = event[:, None] & (time_[:, None] < time_[None, :])
    if comparable.any():
        conc = comparable * ((risk[:, None] > risk[None, :]) + 0.5 * (risk[:, None] == risk[None, :]))
        den = ((w @ comparable.astype(float)) * w).sum(1)
        cidx = np.where(den > 0, ((w @ conc) * w).sum(1) / np.where(den > 0, den, 1), np.nan)
    else:
        cidx = np.full(d, np.nan)
    return auc, cidx


def concordance_point(time_, event, risk) -> float:
    if event.sum() < 1 or len(time_) < 5:
        return np.nan
    comparable = event[:, None] & (time_[:, None] < time_[None, :])
    if not comparable.any():
        return np.nan
    conc = comparable * ((risk[:, None] > risk[None, :]) + 0.5 * (risk[:, None] == risk[None, :]))
    return float(conc.sum() / comparable.sum())


def validate(cells, universes, n_check: int = 80) -> None:
    """Check the vectorised observed-data values against sklearn and scikit-survival."""
    rng = np.random.default_rng(1)
    worst_auc = worst_c = 0.0
    for i in rng.choice(len(cells), size=min(n_check, len(cells)), replace=False):
        c = cells[i]
        canc = c["cancer"][0] if c["scope"] == "per_cancer" else rng.choice(np.unique(c["cancer"]))
        sel = c["cancer"] == canc
        u = universes[(c["endpoint"], canc)]
        idx = np.array([u["index"][p] for p in c["patient"][sel]])
        auc, cidx = weighted_metrics(u["W"][:1], idx, c["y"][sel], c["risk"][sel], c["time"][sel], c["event"][sel])
        if len(np.unique(c["y"][sel])) == 2:
            worst_auc = max(worst_auc, abs(auc[0] - roc_auc_score(c["y"][sel], c["risk"][sel])))
        if c["event"][sel].sum() >= 2 and (~c["event"][sel]).sum() >= 1 and not np.isnan(cidx[0]):
            ref = concordance_index_censored(c["event"][sel], c["time"][sel], c["risk"][sel])[0]
            worst_c = max(worst_c, abs(cidx[0] - ref))
    print(f"validation over {n_check} test-set subsets: max |AUC - sklearn| = {worst_auc:.2e}; max |C - scikit-survival| = {worst_c:.2e}")
    assert worst_auc < 1e-9, "AUC implementation disagrees with sklearn"
    assert worst_c < 5e-3, "C-index implementation disagrees with scikit-survival"


# ---------------------------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics["roc_auc"].notna()]
    cells = load_cells(metrics)
    print(f"loaded {len(cells)} saved test sets in {time.time() - t0:.0f}s")
    universes = build_universes(cells)
    validate(cells, universes)

    # group -> list of per-test-set metric vectors (length N_BOOT + 1)
    auc_groups: dict[tuple, list] = defaultdict(list)
    cidx_groups: dict[tuple, list] = defaultdict(list)
    allpairs_c: dict[tuple, list] = defaultdict(list)
    for c in cells:
        if c["scope"] == "pooled":
            allpairs_c[(c["endpoint"], c["model"])].append(concordance_point(c["time"], c["event"], c["risk"]))
        for canc in np.unique(c["cancer"]):
            sel = c["cancer"] == canc
            u = universes[(c["endpoint"], canc)]
            idx = np.array([u["index"][p] for p in c["patient"][sel]])
            a, ci = weighted_metrics(u["W"], idx, c["y"][sel], c["risk"][sel], c["time"][sel], c["event"][sel])
            key = (c["endpoint"], c["model"], c["scope"], canc)
            auc_groups[key].append(a)
            cidx_groups[key].append(ci)
    print(f"scored all test sets under {N_BOOT} bootstrap draws in {time.time() - t0:.0f}s")

    def collapse(groups):
        return {k: np.nanmean(np.stack(v), axis=0) for k, v in groups.items()}

    summary_rows, cancer_rows, did_rows, harm_rows, table_rows = [], [], [], [], []
    for metric, groups in (("auc", collapse(auc_groups)), ("cidx", collapse(cidx_groups))):
        endpoints = sorted({k[0] for k in groups})
        for e in endpoints:
            models = sorted({k[1] for k in groups if k[0] == e})
            delta = {}
            for m in models:
                cancers = sorted({k[3] for k in groups if k[:2] == (e, m) and k[2] == "per_cancer"} &
                                 {k[3] for k in groups if k[:2] == (e, m) and k[2] == "pooled"})
                if not cancers:
                    continue
                spec = np.stack([groups[(e, m, "per_cancer", c)] for c in cancers])
                pool = np.stack([groups[(e, m, "pooled", c)] for c in cancers])
                d = pool - spec
                delta[m] = d.mean(axis=0)
                lo, hi = np.nanpercentile(delta[m][1:], [2.5, 97.5])
                summary_rows.append(dict(metric=metric, endpoint=e, model_name=m, n_cancers=len(cancers),
                                         mean_specific=spec[:, 0].mean(), mean_pooled=pool[:, 0].mean(),
                                         mean_delta=delta[m][0], ci_low=lo, ci_high=hi))
                harm_rows.append(dict(metric=metric, endpoint=e, model_name=m, n_cancers=len(cancers),
                                      n_harmed=int((d[:, 0] < 0).sum())))
                for j, c in enumerate(cancers):
                    clo, chi = np.nanpercentile(d[j, 1:], [2.5, 97.5])
                    cancer_rows.append(dict(metric=metric, endpoint=e, model_name=m, cancer=c, specific=spec[j, 0],
                                            pooled=pool[j, 0], delta=d[j, 0], ci_low=clo, ci_high=chi))
                if metric == "cidx":
                    table_rows.append(dict(endpoint=e, model_name=m, c_specific_within=spec[:, 0].mean(),
                                           c_pooled_within=pool[:, 0].mean(),
                                           c_pooled_allpairs=float(np.nanmean(allpairs_c[(e, m)])) if allpairs_c.get((e, m)) else np.nan))
            fm = [m for m in FOUNDATION if m in delta]
            tr = [m for m in TREES if m in delta]
            fm_d = np.mean([delta[m] for m in fm], axis=0)
            tr_d = np.mean([delta[m] for m in tr], axis=0)
            did = fm_d - tr_d
            lo, hi = np.nanpercentile(did[1:], [2.5, 97.5])
            did_rows.append(dict(metric=metric, endpoint=e, n_foundation=len(fm), foundation_models=";".join(fm),
                                 foundation_mean_delta=fm_d[0], tree_mean_delta=tr_d[0], did=did[0],
                                 did_ci_low=lo, did_ci_high=hi, p_did_ge_0=float((did[1:] >= 0).mean())))

    pd.DataFrame(summary_rows).to_csv(OUT / "foldaware_pooling_summary.csv", index=False)
    pd.DataFrame(cancer_rows).to_csv(OUT / "foldaware_pooling_per_cancer.csv", index=False)
    pd.DataFrame(did_rows).to_csv(OUT / "foldaware_pooling_did.csv", index=False)
    pd.DataFrame(harm_rows).to_csv(OUT / "foldaware_pooling_harmed.csv", index=False)

    table = pd.DataFrame(table_rows)
    auc_pool_all = metrics[metrics.scope == "pooled"].groupby(["endpoint", "model_name"])["roc_auc"].mean().rename("auc_pooled_allpairs")
    auc_sum = pd.DataFrame(summary_rows)
    auc_sum = auc_sum[auc_sum.metric == "auc"].set_index(["endpoint", "model_name"])[["mean_specific", "mean_pooled"]]
    auc_sum.columns = ["auc_specific_within", "auc_pooled_within"]
    table = table.set_index(["endpoint", "model_name"]).join(auc_sum).join(auc_pool_all).reset_index()
    table.to_csv(OUT / "cindex_eligible_summary.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("\nFamily difference (foundation minus trees), change after pooling:")
        print(pd.DataFrame(did_rows).drop(columns=["foundation_models"]).round(3).to_string(index=False))
        s = pd.DataFrame(summary_rows)
        print("\nSelected models, change in within-cancer metric after pooling (95% CI):")
        for metric in ("auc", "cidx"):
            for e in ("os_5yr", "extreme_os"):
                for m in ("tabfm_default", "tabpfn_v3", "random_forest", "catboost"):
                    r = s[(s.metric == metric) & (s.endpoint == e) & (s.model_name == m)].iloc[0]
                    print(f"  {metric:4s} {e:10s} {m:14s} {r.mean_delta:+.3f} ({r.ci_low:+.3f}, {r.ci_high:+.3f})")
        print("\nAgreement of rank order between AUC and C across models (Spearman):")
        for e in ("os_3yr", "os_5yr", "extreme_os"):
            t = table[table.endpoint == e].dropna(subset=["c_pooled_allpairs"])
            print(f"  {e}: cancer-specific within {spearmanr(t.auc_specific_within, t.c_specific_within)[0]:.2f}, "
                  f"pooled all pairs {spearmanr(t.auc_pooled_allpairs, t.c_pooled_allpairs)[0]:.2f}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
