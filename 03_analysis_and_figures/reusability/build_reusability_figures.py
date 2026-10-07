"""Main figures for the reusability-report version of the manuscript.

Reads saved results and the CSVs written by compute_reusability_statistics.py; fits nothing.
Writes paper/figures/reusability/fig{1..5}_*.{pdf,png} and per-figure source tables. The
original manuscript figures in paper/figures/manuscript/ are not touched.
"""

from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

PAPER = REPO
OLD_SRC = RESULTS / "source_data/manuscript"
TABLES = RESULTS / "tables"
OUT = RESULTS / "figures/reusability"
SRC = RESULTS / "source_data/reusability"

# Palette: validated categorical slots (blue, orange, aqua, violet) + neutral ink.
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE, NEUTRAL = "#e4e3df", "#ffffff", "#f0efec"
FAMILY_COLOR = {"Linear": INK_3, "Tree ensemble": "#eb6834", "AutoML": "#1baf7a",
                "TabFM": "#4a3aa7", "TabPFN": "#2a78d6"}
TABPFN_RAMP = {"tabpfn_v1": "#86b6ef", "tabpfn_v2": "#5598e7", "tabpfn_v2_5": "#2a78d6",
               "tabpfn_v2_6": "#1c5cab", "tabpfn_v3": "#0d366b"}
MODEL_ORDER = ["logistic_regression", "random_forest", "catboost", "xgboost", "lightgbm", "autogluon",
               "tabfm_default", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3"]
LABEL = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "catboost": "CatBoost",
         "xgboost": "XGBoost", "lightgbm": "LightGBM", "autogluon": "AutoGluon", "tabfm_default": "TabFM",
         "tabfm": "TabFM", "tabpfn": "TabPFN", "tabpfn_v1": "TabPFN v1", "tabpfn_v2": "TabPFN v2",
         "tabpfn_v2_5": "TabPFN v2.5", "tabpfn_v2_6": "TabPFN v2.6", "tabpfn_v3": "TabPFN v3"}
FAMILY = {"logistic_regression": "Linear", "random_forest": "Tree ensemble", "catboost": "Tree ensemble",
          "xgboost": "Tree ensemble", "lightgbm": "Tree ensemble", "autogluon": "AutoML",
          "tabfm_default": "TabFM", "tabfm": "TabFM"}
ENDPOINTS = ["os_3yr", "os_5yr", "extreme_os"]
ENDPOINT_LABEL = {"os_3yr": "3-year survival", "os_5yr": "5-year survival", "extreme_os": "Extreme survival"}
CANCERS = ["BRCA", "ESCA", "HNSCC", "LSCC", "LUAD"]


def color(model: str) -> str:
    if model in TABPFN_RAMP:
        return TABPFN_RAMP[model]
    if model == "tabpfn":
        return FAMILY_COLOR["TabPFN"]
    return FAMILY_COLOR[FAMILY[model]]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK_3,
        "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def letter(ax: plt.Axes, text: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def card(ax, x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
                                facecolor=SURFACE, edgecolor=edge, linewidth=1.2, transform=ax.transAxes))
    ax.text(x + 0.014, y + h - 0.09, title, transform=ax.transAxes, fontsize=7.6, fontweight="bold", color=edge, va="top")
    ax.text(x + 0.014, y + h - 0.34, body, transform=ax.transAxes, fontsize=6.2, color=INK_2, va="top", linespacing=1.3)


# ----------------------------------------------------------------------------------------------
def figure_1() -> None:
    cohort = pd.read_csv(TABLES / "table_01_cohort_characteristics.csv")
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 1.0], hspace=0.28, wspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    letter(ax, "a", x=-0.02, y=1.0)
    # Keep explanatory labels as vector text, readable at journal column width.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cohort_colors = ["#2a78d6", "#159d92", "#e7a12d", "#747b85", "#d96545"]
    centers = [0.105, 0.365, 0.625, 0.885]
    titles = ["1  Reproduce", "2  Within cancer", "3  Pool cancers", "4  Score within\ndiagnosis"]
    for x, title in zip(centers, titles):
        ax.text(x, 0.98, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK)
    for x in [0.225, 0.485, 0.745]:
        ax.annotate("", xy=(x + 0.028, 0.48), xytext=(x - 0.012, 0.48),
                    arrowprops=dict(arrowstyle="-|>", color=INK_3, lw=1.2))

    # Small overlapping benchmark tables, deliberately without measured values.
    for k in range(3):
        x, y = 0.035 + k * 0.023, 0.49 - k * 0.055
        ax.add_patch(plt.Rectangle((x, y), 0.10, 0.23, fc=SURFACE, ec="#2a78d6", lw=0.8))
        for row in range(4):
            for col in range(4):
                ax.add_patch(plt.Rectangle((x + 0.004 + col * 0.024, y + 0.007 + row * 0.053),
                                          0.020, 0.041, fc="#cbe0f5", ec="none"))
    ax.text(centers[0], 0.23, "Original benchmarks", ha="center", fontsize=8)
    ax.text(centers[0], 0.08, "Matched model\ncomparisons", ha="center", fontsize=8, color=INK_2)

    # Cohort colors are preserved through pooling and subgroup scoring.
    for k, (name, c) in enumerate(zip(CANCERS, cohort_colors)):
        y = 0.68 - k * 0.103
        ax.text(0.29, y, name, fontsize=8, va="center", ha="left")
        ax.add_patch(plt.Rectangle((0.375, y - 0.026), 0.07, 0.052, fc=c, ec="none"))
    ax.text(centers[1], 0.08, "Train separately\nfor each diagnosis", ha="center", fontsize=8, color=INK_2)

    for row, c in enumerate(cohort_colors):
        for col in range(6):
            missing = (row + col) % 5 == 0
            ax.add_patch(plt.Rectangle((0.553 + col * 0.024, 0.68 - row * 0.073),
                                      0.022, 0.063, fc=SURFACE if missing else c,
                                      ec="#b8bdc1", lw=0.35))
    ax.text(centers[2], 0.23, "One combined cohort", ha="center", fontsize=8)
    ax.text(centers[2], 0.08, "Cancer identity and\nzero-pattern controls", ha="center", fontsize=8, color=INK_2)

    ax.text(0.828, 0.76, "Separate", ha="center", fontsize=7.5)
    ax.text(0.937, 0.76, "Pooled", ha="center", fontsize=7.5)
    for k, c in enumerate(cohort_colors):
        y = 0.68 - k * 0.073
        for x in [0.803, 0.912]:
            ax.add_patch(plt.Rectangle((x, y - 0.023), 0.05, 0.046, fc=c, ec="none"))
        ax.plot([0.853, 0.912], [y, y], color=INK_3, lw=0.8)
    ax.text(centers[3], 0.23, "Same patients", ha="center", fontsize=8)
    ax.text(centers[3], 0.08, "Paired scoring\nwithin each cancer", ha="center", fontsize=8, color=INK_2)

    ax = fig.add_subplot(gs[1, 0])
    letter(ax, "b")
    cols = [("n_raw", "All patients", "#d6d5d0"), ("n_os_3yr", "3-year eligible", "#8a8984"),
            ("n_os_5yr", "5-year eligible", "#52514e"), ("n_extreme", "Extreme eligible", "#1f1e1c")]
    y = np.arange(len(cohort))
    h = 0.2
    for k, (col, lab, c) in enumerate(cols):
        vals = pd.to_numeric(cohort[col], errors="coerce")
        ax.barh(y + (k - 1.5) * h, vals, height=h - 0.03, color=c, label=lab)
    ax.set_yticks(y, cohort["cancer"])
    ax.invert_yaxis()
    ax.set_xlabel("Patients")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.0, handlelength=1.0, columnspacing=0.8)
    for yy, n in zip(y, cohort["n_raw"]):
        ax.text(n + 20, yy - 1.5 * h, f"{n:,}", va="center", fontsize=6.3, color=INK_2)
    ax.set_xlim(0, 1450)

    ax = fig.add_subplot(gs[1, 1])
    letter(ax, "c")
    ax.set_axis_off()
    ax.set_xlim(-0.3, 6.6)
    ax.set_ylim(-0.2, 4.4)
    ax.plot([0, 6.3], [3.6, 3.6], color=INK_3, lw=1.2)
    for t, lab in [(0, "Diagnosis"), (3, "3 y"), (5, "5 y")]:
        ax.plot([t, t], [3.45, 3.75], color=INK_2, lw=1.0)
        ax.text(t, 3.95, lab, ha="center", fontsize=6.8, color=INK)
    rows = [("Death ≤ horizon", "event (1)", "#eb6834", 2.7, 2.4), ("Alive > horizon", "non-event (0)", "#2a78d6", 2.0, 5.6),
            ("Censored < horizon", "excluded", INK_3, 1.3, 1.9)]
    for name, cls, c, yy, end in rows:
        ax.plot([0, end], [yy, yy], color=c, lw=2.2, solid_capstyle="round")
        marker = "x" if c != INK_3 else "o"
        ax.plot(end, yy, marker=marker, color=c, ms=5, mfc=c if marker == "x" else SURFACE, mew=1.4)
        ax.text(0, yy + 0.22, f"{name} → {cls}", fontsize=6.5, color=INK)
    ax.text(0, 0.35, "Extreme survival: death ≤ 3 y vs alive > 5 y;\nintermediate patients excluded.", fontsize=6.3, color=INK_2)

    ax = fig.add_subplot(gs[1, 2])
    letter(ax, "d")
    ax.set_axis_off()
    parts = [("Train", 0.64, "#d6d5d0", "fit; features chosen here"), ("Validation", 0.16, "#8a8984", "threshold only"),
             ("Test", 0.20, "#1f1e1c", "scored once")]
    x0 = 0.0
    for name, frac, c, note in parts:
        ax.add_patch(plt.Rectangle((x0, 0.62), frac - 0.01, 0.16, facecolor=c, edgecolor="none", transform=ax.transAxes))
        ax.text(x0 + (frac - 0.01) / 2, 0.70, f"{int(frac * 100)}%", transform=ax.transAxes, ha="center", va="center",
                fontsize=6.8, color=INK if c == "#d6d5d0" else SURFACE, fontweight="bold")
        ax.text(x0 + (frac - 0.01) / 2, 0.84, {"Validation": "Val."}.get(name, name), transform=ax.transAxes,
                fontsize=6.8, color=INK, fontweight="bold", ha="center")
        x0 += frac
    ax.text(0, 0.50, "Pooled 3-year task: ~1,050 patients fitted,\n~330 tested per split; all 1,645 scored once\nper repeat (5 repeats × 5 folds).",
            transform=ax.transAxes, fontsize=6.3, color=INK_2, va="top")
    ax.text(0, 0.16, "Patient-grouped; identical for every model;\nfrozen with checksums before any run.",
            transform=ax.transAxes, fontsize=6.3, color=INK_2, va="top")
    save(fig, "fig1_roadmap_and_design")


# ----------------------------------------------------------------------------------------------
def figure_2() -> None:
    rep = pd.read_csv(SRC / "reproduction_auc.csv")
    base = rep[rep["study"] == "baselines"].pivot(index="dataset", columns="model", values="roc_auc").dropna()
    gen = rep[rep["study"] == "generations"].pivot(index="dataset", columns="model", values="roc_auc").dropna()
    base.to_csv(SRC / "fig2_baseline_auc.csv")
    gen.to_csv(SRC / "fig2_generation_auc.csv")

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.1], wspace=0.95)

    ax = fig.add_subplot(gs[0])
    letter(ax, "a")
    order = base["tabpfn"].sort_values().index
    others = [c for c in base.columns if c != "tabpfn"]
    for i, ds in enumerate(order):
        row = base.loc[ds]
        ax.plot([row[others].min(), row[others].max()], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        for m in others:
            ax.scatter(row[m], i, s=14, color=color(m), alpha=0.85, zorder=2, linewidths=0)
        ax.scatter(row["tabpfn"], i, s=40, color=SURFACE, edgecolor=FAMILY_COLOR["TabPFN"], linewidth=1.6, zorder=3)
        rank = int((row.round(3) > round(row["tabpfn"], 3)).sum()) + 1
        ax.text(1.035, i, f"#{rank}", va="center", ha="left", fontsize=6.3, color=INK_2)
    display = {"chum": "churn", "credit_g": "credit-g", "blood_transfusion_service_center": "blood transfusion"}
    ax.set_yticks(range(len(order)), [display.get(d, d) for d in order])
    ax.set_xlim(0.68, 1.075)
    ax.set_xlabel("Test ROC AUC")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=FAMILY_COLOR["TabPFN"], mew=1.6, ms=6, label="TabPFN"),
               Line2D([], [], marker="o", ls="", color=FAMILY_COLOR["Tree ensemble"], ms=4, label="Tree ensembles"),
               Line2D([], [], marker="o", ls="", color=FAMILY_COLOR["AutoML"], ms=4, label="AutoGluon"),
               Line2D([], [], marker="o", ls="", color=INK_3, ms=4, label="Logistic regression")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=6.0, handletextpad=0.3, columnspacing=0.8)

    ax = fig.add_subplot(gs[1])
    letter(ax, "b")
    means = base.mean().sort_values()
    for i, (m, v) in enumerate(means.items()):
        ax.scatter(base[m], [i] * len(base), s=8, color=color(m), alpha=0.35, linewidths=0)
        ax.scatter(v, i, s=34, color=color(m), zorder=3, edgecolor=SURFACE, linewidth=0.8)
        ax.text(v, i + 0.32, f"{v:.3f}", ha="center", fontsize=6.0, color=INK_2)
    ax.set_yticks(range(len(means)), [LABEL[m] for m in means.index])
    ax.set_xlabel("ROC AUC (6 datasets)")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(gs[2])
    letter(ax, "c")
    cols = [c for c in ["tabpfn_v1", "tabpfn_v2", "tabpfn_v2_5", "tabpfn_v2_6", "tabpfn_v3", "tabfm"] if c in gen.columns]
    rel = gen[cols].sub(gen[cols].mean(axis=1), axis=0)
    x = np.arange(len(cols))
    for ds, row in rel.iterrows():
        ax.plot(x, row.to_numpy(), color=GRID, lw=0.9, zorder=1)
    for i, c in enumerate(cols):
        ax.scatter([i] * len(rel), rel[c], s=10, color=color(c), alpha=0.6, linewidths=0, zorder=2)
        ax.scatter(i, rel[c].mean(), s=36, color=color(c), edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.axhline(0, color=INK_3, lw=0.7, ls="--")
    ax.set_xticks(x, [LABEL[c].replace("TabPFN ", "") for c in cols], fontsize=6.3)
    ax.set_ylim(-0.07, 0.045)
    ax.set_ylabel("AUC minus dataset mean")
    ax.set_xlabel("TabPFN generation / TabFM")
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_reproduction")


# ----------------------------------------------------------------------------------------------
def figure_3() -> None:
    ranks = pd.read_csv(SRC / "within_cancer_ranks.csv")
    fried = pd.read_csv(SRC / "within_cancer_friedman.csv").iloc[0]
    delta = pd.read_csv(SRC / "paired_delta_vs_rf.csv")

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.45, 1.0], hspace=0.6, wspace=0.42)

    ax = fig.add_subplot(gs[0, :])
    letter(ax, "a", x=-0.07)
    task = ranks.pivot_table(index="model_name", columns=["endpoint", "cancer"], values="roc_auc")
    task = task.reindex(MODEL_ORDER)[[c for e in ENDPOINTS for c in task.columns if c[0] == e]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ink", ["#f7f6f3", "#c9c8c3", "#8a8984", "#3b3a37"])
    im = ax.imshow(task.to_numpy(), cmap=cmap, vmin=0.40, vmax=0.72, aspect="auto")
    best = task.to_numpy().argmax(axis=0)
    for i in range(task.shape[0]):
        for j in range(task.shape[1]):
            v = task.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color=SURFACE if v > 0.61 else INK, fontweight="bold" if i == best[j] else "normal")
    ax.set_yticks(range(len(task)), [LABEL[m] for m in task.index])
    for tick, m in zip(ax.get_yticklabels(), task.index):
        tick.set_color(color(m))
    ax.set_xticks(range(task.shape[1]), [c[1] for c in task.columns], rotation=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    edges = np.cumsum([sum(1 for c in task.columns if c[0] == e) for e in ENDPOINTS])
    start = 0
    for e, end in zip(ENDPOINTS, edges):
        ax.text((start + end - 1) / 2, -1.05, ENDPOINT_LABEL[e], ha="center", fontsize=7, color=INK, fontweight="bold")
        if end < task.shape[1]:
            ax.axvline(end - 0.5, color=SURFACE, lw=3)
        start = end
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("ROC AUC", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    task.to_csv(SRC / "fig3a_task_auc.csv")

    ax = fig.add_subplot(gs[1, 0])
    letter(ax, "b", x=-0.1)
    models = [m for m in MODEL_ORDER if m != "random_forest"]
    data = [delta.loc[delta["model_name"] == m, "delta_vs_rf"].dropna().to_numpy() for m in models]
    parts = ax.violinplot(data, vert=False, widths=0.8, showextrema=False)
    for body, m in zip(parts["bodies"], models):
        body.set_facecolor(color(m))
        body.set_alpha(0.35)
        body.set_edgecolor("none")
    for i, (m, d) in enumerate(zip(models, data), start=1):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=color(m), lw=2.2, solid_capstyle="round")
        ax.scatter(med, i, s=16, color=SURFACE, edgecolor=color(m), linewidth=1.2, zorder=3)
    ax.axvline(0, color=INK_3, lw=0.8, ls="--")
    ax.set_yticks(range(1, len(models) + 1), [LABEL[m] for m in models])
    ax.set_xlim(-0.35, 0.35)
    ax.set_xlabel("ROC AUC minus random forest (same test set)")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(gs[1, 1])
    letter(ax, "c", x=-0.16)
    mean_rank = ranks.groupby("model_name")["rank"].mean().sort_values()
    for i, (m, r) in enumerate(mean_rank.items()):
        ax.plot([1, r], [i, i], color=GRID, lw=1.0, zorder=1)
        ax.scatter(r, i, s=34, color=color(m), edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.text(r + 0.25, i, f"{r:.1f}", va="center", fontsize=6.0, color=INK_2)
    cd = fried["nemenyi_cd_alpha05"]
    y_cd = -1.1
    ax.plot([mean_rank.iloc[0], mean_rank.iloc[0] + cd], [y_cd, y_cd], color=INK, lw=1.6)
    for xe in (mean_rank.iloc[0], mean_rank.iloc[0] + cd):
        ax.plot([xe, xe], [y_cd - 0.25, y_cd + 0.25], color=INK, lw=1.0)
    ax.text(mean_rank.iloc[0] + cd + 0.25, y_cd, f"critical difference {cd:.2f}", va="center", fontsize=6.0, color=INK)
    ax.set_yticks(range(len(mean_rank)), [LABEL[m] for m in mean_rank.index])
    ax.invert_yaxis()
    ax.set_ylim(len(mean_rank) + 0.4, -1.8)
    ax.set_xlim(1, 11)
    ax.set_xlabel("Mean rank (1 = best)")
    ax.text(0.98, 0.02, f"Friedman χ² = {fried['chi2']:.1f}, P = {fried['p_value']:.3f}", transform=ax.transAxes,
            ha="right", fontsize=6.0, color=INK_2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_within_cancer")


# ----------------------------------------------------------------------------------------------
def figure_4() -> None:
    metrics = pd.read_csv(OLD_SRC / "model_fold_metrics.csv")
    shortcut = pd.read_csv(OLD_SRC / "figure_04_shortcut_controls.csv")
    perm = pd.read_csv(OLD_SRC / "figure_04_permutation_distributions.csv")
    held = pd.read_csv(TABLES / "table_03_shortcut_summary.csv").set_index("endpoint")
    sep = pd.read_csv(OLD_SRC / "figure_04_cohort_separability.csv").set_index("control")
    scope_mean = metrics.groupby(["scope", "endpoint", "model_name"])["roc_auc"].mean().unstack(0)

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], hspace=0.62, wspace=0.5)

    for k, e in enumerate(ENDPOINTS):
        ax = fig.add_subplot(gs[0, k])
        if k == 0:
            letter(ax, "a")
        ax.set_title(ENDPOINT_LABEL[e], pad=6)
        sub = scope_mean.loc[e].dropna()
        for m in MODEL_ORDER:
            if m not in sub.index:
                continue
            a, b = sub.loc[m, "per_cancer"], sub.loc[m, "pooled"]
            lw = 1.6 if FAMILY.get(m) != "Linear" else 1.0
            ax.plot([0, 1], [a, b], color=color(m), lw=lw, alpha=0.9, marker="o", ms=3.2)
        ax.set_xticks([0, 1], ["Within\ncancer", "Pooled"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0.48, 0.81)
        ax.axhline(0.5, color=INK_3, lw=0.6, ls=":")
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_ylabel("Mean ROC AUC")
        if k == 1:
            handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=1.8, label=f) for f in ["TabPFN", "TabFM", "Tree ensemble", "AutoML", "Linear"]]
            ax.legend(handles=handles, loc="upper left", fontsize=5.9, handlelength=1.2)
    scope_mean.to_csv(SRC / "fig4a_scope_means.csv")

    ax = fig.add_subplot(gs[1, :2])
    letter(ax, "b", x=-0.07)
    rows = []
    for e in ENDPOINTS:
        pooled = scope_mean.loc[e, "pooled"].dropna()
        trees = pooled[[m for m in pooled.index if FAMILY.get(m) == "Tree ensemble"]]
        fms = pooled[[m for m in pooled.index if m.startswith("tab")]]
        sc = shortcut[shortcut["endpoint"] == e].set_index("control")["roc_auc"]
        within_perm = perm[(perm["endpoint"] == e) & (perm["permutation"] == "within_cancer")]["roc_auc"]
        rows += [
            (e, "Best foundation model", fms.max(), FAMILY_COLOR["TabPFN"], "benchmark"),
            (e, "Best tree ensemble", trees.max(), FAMILY_COLOR["Tree ensemble"], "benchmark"),
            (e, "Cancer type only", sc["cancer_identity_only"], "#3b3a37", "historical"),
            (e, "Zero pattern only", sc["structural_zero_pattern"], "#77766f", "historical"),
            (e, "Labels shuffled within cancer", within_perm.mean(), "#a9a8a2", "historical"),
            (e, "Held-out cancer (linear)", held.loc[e, "heldout_linear_auc"], "#d6d5d0", "historical"),
        ]
    table = pd.DataFrame(rows, columns=["endpoint", "comparator", "roc_auc", "color", "setting"])
    table.drop(columns="color").to_csv(SRC / "fig4b_controls.csv", index=False)
    n_comp = 6
    width = 0.13
    for i, e in enumerate(ENDPOINTS):
        sub = table[table["endpoint"] == e].reset_index(drop=True)
        for j, r in sub.iterrows():
            x = i + (j - (n_comp - 1) / 2) * width
            hatch = "////" if r["setting"] == "historical" else None
            ax.bar(x, r["roc_auc"] - 0.4, bottom=0.4, width=width - 0.015, color=r["color"],
                   edgecolor=SURFACE if hatch is None else SURFACE, hatch=hatch, linewidth=0)
            ax.text(x, r["roc_auc"] + 0.006, f"{r['roc_auc']:.2f}", ha="center", fontsize=5.4, color=INK_2, rotation=90)
    ax.axhline(0.5, color=INK_3, lw=0.7, ls="--")
    ax.set_xticks(range(3), [ENDPOINT_LABEL[e] for e in ENDPOINTS])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.grid(axis="x", visible=False)
    legend_items = table[table["endpoint"] == ENDPOINTS[0]]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=r["color"], hatch="////" if r["setting"] == "historical" else None,
                             edgecolor=SURFACE, label=r["comparator"]) for _, r in legend_items.iterrows()]
    ax.legend(handles=handles, ncol=3, loc="upper left", fontsize=5.9, handlelength=1.3, columnspacing=1.0)
    ax.text(1.0, -0.2, "Hatched: single historical split (n_test 141–247), shown for context, not a matched comparison.",
            transform=ax.transAxes, ha="right", fontsize=5.8, color=INK_3)

    ax = fig.add_subplot(gs[1, 2])
    letter(ax, "c", x=-0.16)
    vals = [("Zero\npattern", sep.loc["structural_zero_pattern", "balanced_accuracy"], INK_2),
            ("Molecular\nvalues", sep.loc["molecular_values", "balanced_accuracy"], "#c9c8c3")]
    for i, (lab, v, c) in enumerate(vals):
        ax.barh(i, v, color=c, height=0.55)
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=6.5, color=INK)
    ax.axvline(0.2, color=INK_3, lw=0.8, ls="--")
    ax.text(0.22, 0.5, "chance (0.20)", fontsize=5.8, color=INK_3, va="center")
    ax.set_yticks([0, 1], [v[0] for v in vals])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Balanced accuracy")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_pooling_and_controls")


# ----------------------------------------------------------------------------------------------
def figure_5() -> None:
    """Fold-aware within-cancer effect of pooled training (fold_aware_contrast.py)."""
    summary = pd.read_csv(SRC / "foldaware_pooling_summary.csv")
    summary = summary[summary["metric"] == "auc"]
    did = pd.read_csv(SRC / "foldaware_pooling_did.csv")
    per = pd.read_csv(SRC / "foldaware_pooling_per_cancer.csv")
    per = per[per["metric"] == "auc"]

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.18)
    all_models = list(MODEL_ORDER)

    for k, e in enumerate(ENDPOINTS):
        ax = fig.add_subplot(gs[0, k])
        ax.set_title(ENDPOINT_LABEL[e], pad=6)
        if k == 0:
            letter(ax, "a", x=-0.62)
        sub = summary[summary["endpoint"] == e].set_index("model_name")
        for i, m in enumerate(all_models):
            if m not in sub.index:
                ax.text(0, i, "not evaluated", ha="center", va="center", fontsize=5.6, color=INK_3)
                continue
            r = sub.loc[m]
            resolved = r["ci_high"] < 0 or r["ci_low"] > 0
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color(m), lw=1.6, solid_capstyle="round")
            ax.scatter(r["mean_delta"], i, s=28, color=color(m) if resolved else SURFACE, edgecolor=color(m), linewidth=1.3, zorder=3)
        ax.axvline(0, color=INK_3, lw=0.8, ls="--")
        ax.set_yticks(range(len(all_models)))
        ax.set_yticklabels([LABEL[m] for m in all_models] if k == 0 else [])
        if k == 0:
            for tick, m in zip(ax.get_yticklabels(), all_models):
                tick.set_color(color(m))
        ax.set_ylim(len(all_models) - 0.5, -0.5)
        ax.set_xlim(-0.13, 0.13)
        ax.grid(axis="y", visible=False)
        if k == 1:
            ax.set_xlabel("Within-cancer ROC AUC: pooled minus cancer-specific training (95% CI)")
    handles = [Line2D([], [], marker="o", ls="", mfc=INK_2, mec=INK_2, ms=5, label="CI excludes 0"),
               Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK_2, ms=5, label="CI includes 0")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, fontsize=6.3)

    ax = fig.add_subplot(gs[1, :2])
    letter(ax, "b", x=-0.07)
    groups = [("Foundation models", lambda m: m.startswith("tab"), FAMILY_COLOR["TabPFN"]),
              ("Tree ensembles", lambda m: FAMILY.get(m) == "Tree ensemble", FAMILY_COLOR["Tree ensemble"])]
    rng = np.random.default_rng(3)
    for i, e in enumerate(ENDPOINTS):
        for j, (name, test, c) in enumerate(groups):
            d = per[(per["endpoint"] == e) & per["model_name"].map(test)]["delta"].to_numpy()
            x = i + (j - 0.5) * 0.32
            ax.scatter(x + rng.uniform(-0.09, 0.09, len(d)), d, s=10, color=c, alpha=0.55, linewidths=0)
            ax.plot([x - 0.12, x + 0.12], [np.median(d)] * 2, color=c, lw=2.2)
    ax.axhline(0, color=INK_3, lw=0.8, ls="--")
    ax.set_xticks(range(3), [ENDPOINT_LABEL[e] for e in ENDPOINTS])
    ax.set_ylabel("Pooled minus cancer-specific")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=4, label=n) for n, _, c in groups]
              + [Line2D([], [], color=INK_2, lw=2, label="median")], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=5.9)

    ax = fig.add_subplot(gs[1, 2])
    letter(ax, "c", x=-0.1)
    for i, e in enumerate(ENDPOINTS):
        for metric, marker, off in (("auc", "o", -0.17), ("cidx", "s", 0.17)):
            r = did[(did["endpoint"] == e) & (did["metric"] == metric)].iloc[0]
            resolved = r["did_ci_high"] < 0 or r["did_ci_low"] > 0
            y = i + off
            ax.plot([r["did_ci_low"], r["did_ci_high"]], [y, y], color=INK, lw=1.4, solid_capstyle="round")
            ax.scatter(r["did"], y, s=26, marker=marker, color=INK if resolved else SURFACE, edgecolor=INK, linewidth=1.2, zorder=3)
            ax.text(r["did_ci_high"] + 0.003, y, f"{r['did']:+.3f}", va="center", fontsize=5.8, color=INK_2)
    ax.axvline(0, color=INK_3, lw=0.8, ls="--")
    ax.set_yticks(range(3), [ENDPOINT_LABEL[e] for e in ENDPOINTS])
    ax.yaxis.tick_right()
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(-0.06, 0.075)
    ax.set_xlabel("Foundation models minus trees (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK_2, mec=INK_2, ms=4.5, label="ROC AUC"),
                       Line2D([], [], marker="s", ls="", mfc=INK_2, mec=INK_2, ms=4.5, label="Harrell C")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.9)
    save(fig, "fig5_within_cancer_effect")


def figure_ed_cindex() -> None:
    """Extended Data: Harrell C-index of the saved horizon-classifier risk scores, eligible patients only."""
    t = pd.read_csv(SRC / "cindex_eligible_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True, gridspec_kw=dict(wspace=0.08))
    labels = ["Specific,\nwithin cancer", "Pooled,\nwithin cancer", "Pooled,\nall pairs"]
    cols = ["c_specific_within", "c_pooled_within", "c_pooled_allpairs"]
    for ax, e in zip(axes, ENDPOINTS):
        ax.set_title(ENDPOINT_LABEL[e], pad=6)
        sub = t[t["endpoint"] == e].set_index("model_name")
        for m in MODEL_ORDER:
            if m not in sub.index:
                continue
            vals = sub.loc[m, cols].to_numpy(float)
            if np.isnan(vals[1]):
                ax.scatter(0, vals[0], s=14, color=color(m), zorder=3, alpha=0.9, linewidths=0)
                continue
            ax.plot(range(3), vals, color=color(m), lw=1.0 if FAMILY.get(m) == "Linear" else 1.6, marker="o", ms=3.2, alpha=0.9)
        ax.axhline(0.5, color=INK_3, lw=0.8, ls=":")
        ax.set_xticks(range(3), labels, fontsize=6.0)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0.48, 0.70)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Harrell C-index (eligible patients)")
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=1.8, label=f) for f in ["TabPFN", "TabFM", "Tree ensemble", "AutoML", "Linear"]]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=5.9, handlelength=1.2, columnspacing=0.9)
    save(fig, "ed_cindex_eligible_patients")


def main() -> None:
    setup()
    SRC.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_ed_cindex()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
