from __future__ import annotations

import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_root in _ReleasePath(__file__).resolve().parents:
    if (_release_root / "tabr1_paths.py").exists():
        _release_sys.path.insert(0, str(_release_root))
        break
from tabr1_paths import REPO, RESULTS, FROZEN, TRAIN_READY


import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = REPO
PAPER = ROOT / "paper"
LOCAL_ROOT = RESULTS / "cancer/local_models"
CLOUD_ROOT = RESULTS / "cancer/cloud_models"
FOLD_ROOT = FROZEN
LANDSCAPE_ROOT = RESULTS / "cancer/landscape"
STRESS_ROOT = RESULTS / "cancer/stress_tests"
CONTROL_ROOT = RESULTS / "cancer/shortcut_controls"
FIGURE_ROOT = RESULTS / "figures/manuscript"
SOURCE_ROOT = RESULTS / "source_data/manuscript"
TABLE_ROOT = RESULTS / "tables"

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
]

MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "catboost",
    "xgboost",
    "lightgbm",
    "autogluon",
    "tabfm_default",
    "tabpfn_v2",
    "tabpfn_v2_5",
    "tabpfn_v2_6",
    "tabpfn_v3",
]

MODEL_LABEL = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "autogluon": "AutoGluon",
    "tabfm_default": "TabFM",
    "tabpfn_v2": "TabPFN v2",
    "tabpfn_v2_5": "TabPFN v2.5",
    "tabpfn_v2_6": "TabPFN v2.6",
    "tabpfn_v3": "TabPFN v3",
}

MODEL_FAMILY = {
    "logistic_regression": "Linear",
    "random_forest": "Tree ensemble",
    "catboost": "Boosted tree",
    "xgboost": "Boosted tree",
    "lightgbm": "Boosted tree",
    "autogluon": "AutoML",
    "tabfm_default": "Foundation model",
    "tabpfn_v2": "Foundation model",
    "tabpfn_v2_5": "Foundation model",
    "tabpfn_v2_6": "Foundation model",
    "tabpfn_v3": "Foundation model",
}

FAMILY_COLOR = {
    "Linear": "#7A6F80",
    "Tree ensemble": "#CC6B4E",
    "Boosted tree": "#D89B36",
    "AutoML": "#426B8A",
    "Foundation model": "#158F82",
}

ENDPOINT_ORDER = ["os_3yr", "os_5yr", "extreme_os"]
ENDPOINT_LABEL = {
    "os_3yr": "3-year OS",
    "os_5yr": "5-year OS",
    "extreme_os": "Extreme OS",
}
ENDPOINT_COLOR = {
    "os_3yr": "#284B63",
    "os_5yr": "#D08C3A",
    "extreme_os": "#B84A5A",
}
CANCER_ORDER = ["BRCA", "ESCA", "HNSCC", "LSCC", "LUAD"]


def setup() -> None:
    for path in (FIGURE_ROOT, SOURCE_ROOT, TABLE_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    local = pd.read_csv(LOCAL_ROOT / "all_fold_model_metrics.csv")
    cloud = pd.read_csv(CLOUD_ROOT / "all_fold_model_metrics.csv")
    local["artifact_source"] = "local"
    cloud["artifact_source"] = "cloud"
    if "status" not in local:
        local["status"] = "success"

    keep = [
        "scope",
        "endpoint",
        "cancer",
        "repeat",
        "fold",
        "model_name",
        "n_train",
        "n_validation",
        "n_test",
        "status",
        "threshold",
        "fit_time_s",
        "predict_time_s",
        *METRICS,
        "artifact_source",
    ]
    combined = pd.concat([local[keep], cloud[keep]], ignore_index=True)
    unavailable = combined["status"].ne("success")
    combined.loc[unavailable, [*METRICS, "threshold", "fit_time_s", "predict_time_s"]] = np.nan
    combined["model"] = combined["model_name"].map(MODEL_LABEL)
    combined["family"] = combined["model_name"].map(MODEL_FAMILY)
    combined["endpoint_label"] = combined["endpoint"].map(ENDPOINT_LABEL)

    public_columns = [c for c in combined.columns if c != "status"]
    combined[public_columns].to_csv(SOURCE_ROOT / "model_fold_metrics.csv", index=False, na_rep="NA")

    summary = (
        combined.groupby(["scope", "endpoint", "model_name", "model", "family"], dropna=False)
        .agg(
            n_configurations=("roc_auc", "size"),
            n_evaluated=("roc_auc", "count"),
            **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
            **{f"{metric}_sd": (metric, "std") for metric in METRICS},
            fit_time_s_median=("fit_time_s", "median"),
            predict_time_s_median=("predict_time_s", "median"),
        )
        .reset_index()
    )
    summary.to_csv(SOURCE_ROOT / "model_summary_by_scope_endpoint.csv", index=False, na_rep="NA")
    return combined, summary


def load_fold_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(FOLD_ROOT / "fold_manifest.csv")
    return manifest


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=color,
        facecolor=f"{color}18",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8, fontweight="bold")


def draw_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle_lines: list[str] | None = None,
    border_color: str = "#2563EB",
    fill_color: str = "#F8FAFC",
    title_color: str = "#0F172A",
    body_color: str = "#334155",
    title_size: float = 7.8,
    body_size: float = 6.8,
    linewidth: float = 1.2,
    radius: float = 0.035,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=border_color,
        facecolor=fill_color,
        zorder=2,
    )
    ax.add_patch(box)

    cx = xy[0] + width / 2.0
    if subtitle_lines and len(subtitle_lines) > 0:
        ty = xy[1] + height * 0.77
        ax.text(cx, ty, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=4)

        sep_y = xy[1] + height * 0.58
        ax.plot(
            [xy[0] + width * 0.10, xy[0] + width * 0.90],
            [sep_y, sep_y],
            color=border_color,
            alpha=0.35,
            linewidth=0.8,
            zorder=3,
        )

        by = xy[1] + height * 0.30
        body_str = "\n".join(subtitle_lines)
        ax.text(cx, by, body_str, ha="center", va="center", fontsize=body_size, color=body_color, linespacing=1.20, zorder=4)
    else:
        cy = xy[1] + height / 2.0
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=4)


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#64748B",
    mutation: float = 10,
    linewidth: float = 1.1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=color,
            zorder=5,
        )
    )


def build_figure_1(manifest: pd.DataFrame) -> None:
    cohort_counts = pd.read_csv(LANDSCAPE_ROOT / "cohort_counts.csv")
    class_balance = pd.read_csv(LANDSCAPE_ROOT / "class_balance.csv")
    modality = pd.read_csv(LANDSCAPE_ROOT / "selected_feature_modalities.csv")

    first = manifest.sort_values(["repeat", "fold"]).drop_duplicates(["scope", "endpoint", "cancer"])
    extreme = first[(first["scope"] == "per_cancer") & (first["endpoint"] == "extreme_os")][
        ["cancer", "n_total"]
    ].rename(columns={"n_total": "n_extreme"})
    counts = cohort_counts.merge(extreme, on="cancer", how="left")
    counts.to_csv(SOURCE_ROOT / "figure_01_cohort_flow.csv", index=False, na_rep="NA")

    class_rows = []
    for _, row in first[first["scope"] == "per_cancer"].iterrows():
        mask = (
            (manifest["scope"] == row["scope"])
            & (manifest["endpoint"] == row["endpoint"])
            & (manifest["cancer"] == row["cancer"])
            & (manifest["repeat"] == row["repeat"])
            & (manifest["fold"] == row["fold"])
        )
        r = manifest.loc[mask].iloc[0]
        positives = int(r["class_1_train"] + r["class_1_validation"] + r["class_1_test"])
        class_rows.extend(
            [
                {"cancer": r["cancer"], "endpoint": r["endpoint"], "class": 0, "count": int(r["n_total"] - positives)},
                {"cancer": r["cancer"], "endpoint": r["endpoint"], "class": 1, "count": positives},
            ]
        )
    classes = pd.DataFrame(class_rows)
    classes.to_csv(SOURCE_ROOT / "figure_01_class_counts.csv", index=False)
    modality.to_csv(SOURCE_ROOT / "figure_01_modality_composition.csv", index=False)

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.08], width_ratios=[1.08, 1.05, 1.18])
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes
    fig.suptitle("Cohort construction and leakage-aware evaluation design", fontweight="bold")

    ax_a.set_title("Data provenance and analysis cohorts")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")

    draw_card(
        ax_a,
        (0.01, 0.57),
        0.25,
        0.36,
        "TCGA",
        ["Clinical records", "Multiomics profiles"],
        border_color="#3B82F6",
        fill_color="#EFF6FF",
        title_color="#1D4ED8",
        title_size=9.2,
        body_size=7.8,
    )
    draw_card(
        ax_a,
        (0.01, 0.09),
        0.25,
        0.36,
        "CPTAC",
        ["Proteogenomics", "Matched histology"],
        border_color="#3B82F6",
        fill_color="#EFF6FF",
        title_color="#1D4ED8",
        title_size=9.2,
        body_size=7.8,
    )

    draw_card(
        ax_a,
        (0.34, 0.24),
        0.29,
        0.52,
        "Harmonization",
        ["Patient-level linkage", "Clinical curation", "Strict eligibility check"],
        border_color="#64748B",
        fill_color="#F8FAFC",
        title_color="#1E293B",
        title_size=9.2,
        body_size=7.6,
    )

    draw_card(
        ax_a,
        (0.70, 0.55),
        0.28,
        0.38,
        "5 Disease Cohorts",
        ["BRCA | ESCA | HNSCC", "LSCC | LUAD (N=182–1206)"],
        border_color="#0D9488",
        fill_color="#F0FDFA",
        title_color="#0F766E",
        title_size=9.2,
        body_size=7.6,
    )
    draw_card(
        ax_a,
        (0.70, 0.09),
        0.28,
        0.38,
        "Prognostic Targets",
        ["3-year survival", "5-year survival", "Extreme survival\n(<3 vs ≥5 years)"],
        border_color="#BE123C",
        fill_color="#FFF1F2",
        title_color="#9F1239",
        title_size=9.2,
        body_size=7.6,
    )

    draw_arrow(ax_a, (0.26, 0.75), (0.33, 0.58))
    draw_arrow(ax_a, (0.26, 0.27), (0.33, 0.43))
    draw_arrow(ax_a, (0.63, 0.56), (0.69, 0.70))
    draw_arrow(ax_a, (0.84, 0.54), (0.84, 0.48))
    panel_label(ax_a, "A")

    ax_b.set_title("Eligibility after outcome definition")
    plot_counts = counts.set_index("cancer").reindex(CANCER_ORDER)
    x = np.arange(len(plot_counts))
    series = [
        ("n_raw", "Raw", "#D4D9DF"),
        ("n_os_3yr", "3-year", ENDPOINT_COLOR["os_3yr"]),
        ("n_os_5yr", "5-year", ENDPOINT_COLOR["os_5yr"]),
        ("n_extreme", "Extreme", ENDPOINT_COLOR["extreme_os"]),
    ]
    width = 0.2
    for idx, (column, label, color) in enumerate(series):
        ax_b.bar(x + (idx - 1.5) * width, plot_counts[column], width, label=label, color=color)
    ax_b.set_xticks(x, plot_counts.index)
    ax_b.tick_params(axis="x", labelsize=10.5)
    ax_b.tick_params(axis="y", labelsize=9.0)
    ax_b.set_ylabel("Patients", fontsize=10.0)
    ax_b.legend(frameon=False, ncol=2, fontsize=8.8)
    panel_label(ax_b, "B")

    ax_c.set_title("Cohort-specific selected modalities")
    mod_pivot = (
        modality.pivot_table(index="cancer", columns="modality", values="count", aggfunc="sum", fill_value=0)
        .reindex(CANCER_ORDER)
        .fillna(0)
    )
    sns.heatmap(mod_pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Selected features"}, ax=ax_c)
    ax_c.set_xlabel("")
    ax_c.set_ylabel("")
    ax_c.tick_params(axis="x", rotation=24)
    panel_label(ax_c, "C")

    ax_d.set_title("Fixed-horizon and extreme outcome definitions")
    ax_d.set_xlim(0, 6.2)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    ax_d.plot([0.35, 5.85], [0.55, 0.55], color="#4A5968", linewidth=2.5)
    for xpos, label in [(0.50, "Diagnosis"), (3.0, "3 years"), (5.1, "5 years")]:
        ax_d.plot([xpos, xpos], [0.45, 0.65], color="#284B63", linewidth=2.5)
        ax_d.text(xpos, 0.70, label, ha="center", fontweight="bold", fontsize=10.0, color="#0F172A")
    ax_d.text(3.1, 0.89, "Extreme survival contrast: Death <3 years vs survival ≥5 years", color="#334155", ha="center", fontsize=9.2, fontweight="bold")
    ax_d.text(3.1, 0.35, "Class 1 (Event): Death observed on or before horizon", color="#BE123C", ha="center", fontsize=9.2, fontweight="bold")
    ax_d.text(3.1, 0.22, "Class 0 (Non-event): Follow-up observed beyond horizon", color="#0D9488", ha="center", fontsize=9.2, fontweight="bold")
    ax_d.text(3.1, 0.08, "Censored before horizon: Excluded from fixed-horizon task", color="#64748B", ha="center", fontsize=8.8, fontstyle="italic")
    panel_label(ax_d, "D")

    ax_e.set_title("Outcome composition by cohort")
    classes["task"] = classes["cancer"] + "\n" + classes["endpoint"].map(ENDPOINT_LABEL)
    task_order = []
    for endpoint in ENDPOINT_ORDER:
        task_order.extend(
            classes.loc[classes["endpoint"] == endpoint, "task"].drop_duplicates().tolist()
        )
    class_pivot = classes.pivot_table(index="task", columns="class", values="count", fill_value=0).reindex(task_order)
    ax_e.barh(np.arange(len(class_pivot)), class_pivot.get(0, 0), color="#158F82", label="No event by definition")
    ax_e.barh(
        np.arange(len(class_pivot)),
        class_pivot.get(1, 0),
        left=class_pivot.get(0, 0),
        color="#B84A5A",
        label="Event / early death",
    )
    ax_e.set_yticks(np.arange(len(class_pivot)), class_pivot.index)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Patients")
    ax_e.legend(frameon=False, loc="lower right")
    panel_label(ax_e, "E")

    ax_f.set_title("Frozen repeated evaluation protocol")
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 1)
    ax_f.axis("off")

    draw_card(
        ax_f,
        (0.04, 0.74),
        0.92,
        0.20,
        "Frozen Patient Groups: 5 Repeats × 5 Outer Folds (25 Splits per Task)",
        ["Patient-grouped & class-stratified  •  Identical frozen partitions across all models"],
        border_color="#2563EB",
        fill_color="#EFF6FF",
        title_color="#1E40AF",
        title_size=7.8,
        body_size=6.8,
        radius=0.025,
    )

    draw_card(
        ax_f,
        (0.03, 0.35),
        0.28,
        0.28,
        "TRAIN (64%)",
        ["Select top 100 features", "Variance ranking on train", "Fit model parameters"],
        border_color="#D97706",
        fill_color="#FFFBEB",
        title_color="#B45309",
        title_size=7.6,
        body_size=6.4,
    )

    draw_card(
        ax_f,
        (0.36, 0.35),
        0.28,
        0.28,
        "VALIDATION (16%)",
        ["Select decision threshold", "Model hyperparameter tuning", "Zero test feedback"],
        border_color="#059669",
        fill_color="#ECFDF5",
        title_color="#047857",
        title_size=7.6,
        body_size=6.4,
    )

    draw_card(
        ax_f,
        (0.69, 0.35),
        0.28,
        0.28,
        "TEST (20%)",
        ["Predict held-out once", "Evaluate ROC & PR AUC", "Log loss & Brier score"],
        border_color="#6366F1",
        fill_color="#EEF2FF",
        title_color="#4338CA",
        title_size=7.6,
        body_size=6.4,
    )

    draw_arrow(ax_f, (0.28, 0.73), (0.17, 0.64))
    draw_arrow(ax_f, (0.50, 0.73), (0.50, 0.64))
    draw_arrow(ax_f, (0.72, 0.73), (0.83, 0.64))

    draw_arrow(ax_f, (0.32, 0.49), (0.35, 0.49))
    draw_arrow(ax_f, (0.65, 0.49), (0.68, 0.49))

    ax_f.text(0.5, 0.20, "Strict leakage-safe discipline across all 400 folds", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0F172A")
    ax_f.text(0.5, 0.11, "Feature selection strictly confined to training data  •  Zero test contamination", ha="center", va="center", fontsize=7.0, color="#475569")
    panel_label(ax_f, "F")

    save_figure(fig, "figure_01_cohort_and_design")


def build_figure_2(metrics: pd.DataFrame, manifest: pd.DataFrame) -> None:
    per = metrics[(metrics["scope"] == "per_cancer") & metrics["roc_auc"].notna()].copy()
    grouped = (
        per.groupby(["model_name", "model", "family"])
        .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"), pr_auc_mean=("pr_auc", "mean"))
        .reset_index()
    )
    grouped.to_csv(SOURCE_ROOT / "figure_02_within_cancer_model_summary.csv", index=False)

    heat = per.groupby(["model_name", "endpoint"])["roc_auc"].mean().unstack()
    heat = heat.reindex(MODEL_ORDER)[ENDPOINT_ORDER]
    heat.to_csv(SOURCE_ROOT / "figure_02_within_cancer_auc_heatmap.csv", na_rep="NA")

    keys = ["endpoint", "cancer", "repeat", "fold"]
    rf = per[per["model_name"] == "random_forest"][keys + ["roc_auc"]].rename(columns={"roc_auc": "rf_roc_auc"})
    paired = per.merge(rf, on=keys, how="inner")
    paired["delta_from_random_forest"] = paired["roc_auc"] - paired["rf_roc_auc"]
    delta = paired.groupby(["model_name", "model", "family"])["delta_from_random_forest"].agg(["mean", "std"]).reset_index()
    delta.to_csv(SOURCE_ROOT / "figure_02_paired_auc_delta.csv", index=False)

    fold_classes = manifest[[*keys, "class_1_test", "n_test"]].copy()
    lift = per.merge(fold_classes, on=keys, how="left", suffixes=("", "_manifest"))
    lift["test_prevalence"] = lift["class_1_test"] / lift["n_test_manifest"]
    lift["pr_auc_above_prevalence"] = lift["pr_auc"] - lift["test_prevalence"]
    lift_summary = lift.groupby(["model_name", "model", "family"])["pr_auc_above_prevalence"].agg(["mean", "std"]).reset_index()
    lift_summary.to_csv(SOURCE_ROOT / "figure_02_pr_auc_lift.csv", index=False)

    runtime = (
        per.assign(total_time_s=per["fit_time_s"] + per["predict_time_s"])
        .groupby(["model_name", "model", "family"])
        .agg(roc_auc=("roc_auc", "mean"), median_time_s=("total_time_s", "median"))
        .reset_index()
    )
    runtime.to_csv(SOURCE_ROOT / "figure_02_runtime_pareto.csv", index=False)

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.05, 1.15])
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.suptitle("Matched within-cancer model evaluation", fontweight="bold")

    ax_a.set_title("Model families and evaluation roles")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    family_rows = [
        ("Linear", "Logistic regression", "reference boundary"),
        ("Tree ensemble", "Random forest", "bagged nonlinear baseline"),
        ("Boosted tree", "CatBoost, XGBoost, LightGBM", "strong tabular baselines"),
        ("AutoML", "AutoGluon", "180 s maximum budget per fold"),
        ("Foundation model", "TabFM; TabPFN v2 to v3", "pretrained in-context inference"),
    ]
    for idx, (family, models, role) in enumerate(family_rows):
        ypos = 0.84 - idx * 0.17
        color = FAMILY_COLOR[family]
        ax_a.add_patch(Rectangle((0.04, ypos - 0.05), 0.035, 0.10, color=color))
        ax_a.text(0.10, ypos + 0.030, family, fontweight="bold", color=color, va="center", fontsize=9.2)
        ax_a.text(0.10, ypos + 0.005, role, color="#64748B", va="center", fontsize=8.0, fontstyle="italic")
        ax_a.text(0.10, ypos - 0.028, models, color="#1E293B", va="center", fontsize=8.6)
    panel_label(ax_a, "A")

    ax_b.set_title("Mean ROC AUC across per-cancer folds")
    plot = grouped.set_index("model_name").reindex(MODEL_ORDER).dropna(subset=["roc_auc_mean"])
    ypos = np.arange(len(plot))
    colors = [FAMILY_COLOR[MODEL_FAMILY[m]] for m in plot.index]
    ax_b.errorbar(plot["roc_auc_mean"], ypos, xerr=plot["roc_auc_sd"], fmt="none", color="#7B8791", alpha=0.7, capsize=2)
    ax_b.scatter(plot["roc_auc_mean"], ypos, c=colors, s=42, zorder=3)
    ax_b.axvline(0.5, color="#7B8791", linestyle="--", linewidth=1)
    ax_b.set_yticks(ypos, [MODEL_LABEL[m] for m in plot.index])
    ax_b.set_xlim(0.35, 0.75)
    ax_b.set_xlabel("ROC AUC, mean +/- SD across folds")
    panel_label(ax_b, "B")

    ax_c.set_title("Endpoint-specific within-cancer ROC AUC")
    heat_plot = heat.rename(index=MODEL_LABEL, columns=ENDPOINT_LABEL)
    sns.heatmap(heat_plot, annot=True, fmt=".3f", cmap="vlag", center=0.5, vmin=0.42, vmax=0.68, cbar_kws={"label": "ROC AUC"}, ax=ax_c)
    ax_c.set_xlabel("")
    ax_c.set_ylabel("")
    panel_label(ax_c, "C")

    ax_d.set_title("Paired difference from random forest")
    delta_plot = delta.set_index("model_name").reindex(MODEL_ORDER).dropna(subset=["mean"])
    ypos = np.arange(len(delta_plot))
    colors = [FAMILY_COLOR[MODEL_FAMILY[m]] for m in delta_plot.index]
    ax_d.errorbar(delta_plot["mean"], ypos, xerr=delta_plot["std"], fmt="none", color="#7B8791", alpha=0.7, capsize=2)
    ax_d.scatter(delta_plot["mean"], ypos, c=colors, s=40, zorder=3)
    ax_d.axvline(0, color="#333333", linewidth=1)
    ax_d.set_yticks(ypos, [MODEL_LABEL[m] for m in delta_plot.index])
    ax_d.set_xlabel("Fold-paired delta ROC AUC, mean +/- SD")
    panel_label(ax_d, "D")

    ax_e.set_title("Precision-recall performance above prevalence")
    lift_plot = lift_summary.set_index("model_name").reindex(MODEL_ORDER).dropna(subset=["mean"])
    ypos = np.arange(len(lift_plot))
    colors = [FAMILY_COLOR[MODEL_FAMILY[m]] for m in lift_plot.index]
    ax_e.barh(ypos, lift_plot["mean"], xerr=lift_plot["std"], color=colors, alpha=0.9, error_kw={"ecolor": "#657786", "capsize": 2})
    ax_e.axvline(0, color="#333333", linewidth=1)
    ax_e.set_yticks(ypos, [MODEL_LABEL[m] for m in lift_plot.index])
    ax_e.set_xlabel("PR AUC minus test prevalence")
    panel_label(ax_e, "E")

    ax_f.set_title("Discrimination-runtime tradeoff")
    runtime_offsets = {
        "tabpfn_v2": (5, 13),
        "tabpfn_v2_5": (5, 5),
        "tabpfn_v2_6": (5, -5),
        "tabpfn_v3": (5, -14),
        "tabfm_default": (5, -2),
    }
    for _, row in runtime.iterrows():
        color = FAMILY_COLOR[row["family"]]
        ax_f.scatter(max(row["median_time_s"], 1e-3), row["roc_auc"], color=color, s=48)
        offset = runtime_offsets.get(row["model_name"], (4, 2))
        ax_f.annotate(row["model"], (max(row["median_time_s"], 1e-3), row["roc_auc"]), xytext=offset, textcoords="offset points", fontsize=6.8)
    ax_f.set_xscale("log")
    ax_f.axhline(0.5, color="#7B8791", linestyle="--", linewidth=1)
    ax_f.set_xlabel("Median fit + prediction time per fold (s, log scale)")
    ax_f.set_ylabel("Mean ROC AUC")
    panel_label(ax_f, "F")

    save_figure(fig, "figure_02_within_cancer_performance")


def prediction_file(row: pd.Series) -> Path:
    base = LOCAL_ROOT if row["artifact_source"] == "local" else CLOUD_ROOT / "tabr1_results"
    return (
        base
        / row["scope"]
        / row["endpoint"]
        / row["cancer"]
        / f"repeat_{int(row['repeat']):02d}_fold_{int(row['fold']):02d}"
        / row["model_name"]
        / "test_predictions.csv"
    )


def load_oof_predictions(metrics: pd.DataFrame, models: list[str], endpoints: list[str]) -> pd.DataFrame:
    rows = metrics[
        (metrics["scope"] == "pooled")
        & metrics["model_name"].isin(models)
        & metrics["endpoint"].isin(endpoints)
        & metrics["roc_auc"].notna()
    ]
    frames = []
    for _, row in rows.iterrows():
        path = prediction_file(row)
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        pred["model_name"] = row["model_name"]
        pred["endpoint"] = row["endpoint"]
        pred["repeat"] = int(row["repeat"])
        pred["fold"] = int(row["fold"])
        frames.append(pred)
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    averaged = (
        predictions.groupby(["endpoint", "model_name", "patient_id", "cancer_type", "y_true"], as_index=False)["probability"]
        .mean()
    )
    averaged["model"] = averaged["model_name"].map(MODEL_LABEL)
    return averaged


def curve_tables(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows = []
    pr_rows = []
    for (endpoint, model_name), group in oof.groupby(["endpoint", "model_name"]):
        if group["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["y_true"], group["probability"])
        precision, recall, _ = precision_recall_curve(group["y_true"], group["probability"])
        roc_value = roc_auc_score(group["y_true"], group["probability"])
        pr_value = average_precision_score(group["y_true"], group["probability"])
        roc_rows.extend(
            {"endpoint": endpoint, "model_name": model_name, "fpr": x, "tpr": y, "roc_auc": roc_value}
            for x, y in zip(fpr, tpr)
        )
        pr_rows.extend(
            {"endpoint": endpoint, "model_name": model_name, "recall": x, "precision": y, "pr_auc": pr_value}
            for x, y in zip(recall, precision)
        )
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def build_figure_3(metrics: pd.DataFrame, summary: pd.DataFrame, manifest: pd.DataFrame, ed_variant: bool = False) -> None:
    # ed_variant=True writes figure_03_pooling_effect_ed.*, whose panel E compares estimators instead of showing the
    # patient-averaged paired effect (used as Extended Data Fig. 3 of the reusability report).
    complete_models = ["random_forest", "catboost", "autogluon", "tabfm_default", "tabpfn_v3"]
    oof = load_oof_predictions(metrics, complete_models, ENDPOINT_ORDER)
    oof.to_csv(SOURCE_ROOT / "figure_03_patient_oof_predictions.csv", index=False)
    roc_data, pr_data = curve_tables(oof)
    roc_data.to_csv(SOURCE_ROOT / "figure_03_roc_curves.csv", index=False)
    pr_data.to_csv(SOURCE_ROOT / "figure_03_pr_curves.csv", index=False)

    scope_auc = summary.pivot_table(index=["model_name", "endpoint"], columns="scope", values="roc_auc_mean").reset_index()
    scope_auc["pooling_delta"] = scope_auc.get("pooled") - scope_auc.get("per_cancer")
    scope_auc.to_csv(SOURCE_ROOT / "figure_03_pooling_shift.csv", index=False, na_rep="NA")

    pooled_heat = summary[summary["scope"] == "pooled"].pivot(index="model_name", columns="endpoint", values="roc_auc_mean")
    pooled_heat = pooled_heat.reindex(MODEL_ORDER)[ENDPOINT_ORDER]
    pooled_heat.to_csv(SOURCE_ROOT / "figure_03_pooled_auc_heatmap.csv", na_rep="NA")

    subgroup_rows = []
    for (endpoint, model_name, cancer), group in oof.groupby(["endpoint", "model_name", "cancer_type"]):
        value = np.nan
        if group["y_true"].nunique() == 2:
            value = roc_auc_score(group["y_true"], group["probability"])
        subgroup_rows.append({"endpoint": endpoint, "model_name": model_name, "cancer": cancer, "roc_auc": value})
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(SOURCE_ROOT / "figure_03_pooled_subgroup_auc.csv", index=False, na_rep="NA")

    task_size = (
        metrics[metrics["roc_auc"].notna()]
        .groupby(["scope", "endpoint", "cancer"])
        .agg(n_total=("n_train", lambda x: np.nan), roc_auc=("roc_auc", "mean"))
        .reset_index()
    )
    size_map = manifest.groupby(["scope", "endpoint", "cancer"])["n_total"].first()
    task_size["n_total"] = [size_map.get((r.scope, r.endpoint, r.cancer), np.nan) for r in task_size.itertuples()]
    task_size.to_csv(SOURCE_ROOT / "figure_03_task_size_auc.csv", index=False)

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.08, 1.12])
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.suptitle("Pooling raises overall discrimination but not ranking within a cancer", fontweight="bold")

    ax_a.set_title("Within-cancer versus pooled ROC AUC")
    markers = {m: marker for m, marker in zip(complete_models, ["o", "s", "D", "^", "P"])}
    data_a = scope_auc[scope_auc["model_name"].isin(complete_models)].dropna(subset=["per_cancer", "pooled"])
    for _, row in data_a.iterrows():
        ax_a.scatter(
            row["per_cancer"],
            row["pooled"],
            color=ENDPOINT_COLOR[row["endpoint"]],
            marker=markers[row["model_name"]],
            s=58,
            edgecolor="white",
            linewidth=0.6,
        )
    ax_a.plot([0.45, 0.82], [0.45, 0.82], color="#7B8791", linestyle="--", linewidth=1)
    ax_a.set_xlim(0.48, 0.64)
    ax_a.set_ylim(0.66, 0.82)
    ax_a.set_xlabel("Within-cancer mean ROC AUC")
    ax_a.set_ylabel("Pooled mean ROC AUC")
    endpoint_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ENDPOINT_COLOR[e], label=ENDPOINT_LABEL[e], markersize=7) for e in ENDPOINT_ORDER]
    model_handles = [Line2D([0], [0], marker=markers[m], color="#555555", linestyle="None", label=MODEL_LABEL[m], markersize=7) for m in complete_models]
    ax_a.legend(handles=endpoint_handles + model_handles, frameon=False, fontsize=8.2, ncol=2, loc="lower right")
    panel_label(ax_a, "A")

    selected_endpoint = "os_3yr"
    ax_b.set_title("Pooled 3-year ROC curves")
    for model_name in complete_models:
        data = roc_data[(roc_data["endpoint"] == selected_endpoint) & (roc_data["model_name"] == model_name)]
        if data.empty:
            continue
        label = f"{MODEL_LABEL[model_name]} ({data['roc_auc'].iloc[0]:.3f})"
        ax_b.plot(data["fpr"], data["tpr"], label=label, color=FAMILY_COLOR[MODEL_FAMILY[model_name]], linewidth=1.5, alpha=0.9)
    ax_b.plot([0, 1], [0, 1], linestyle="--", color="#7B8791", linewidth=1)
    ax_b.set_xlabel("False-positive rate")
    ax_b.set_ylabel("True-positive rate")
    ax_b.legend(frameon=False, loc="lower right", fontsize=8.5)
    panel_label(ax_b, "B")

    ax_c.set_title("Pooled 3-year precision-recall curves")
    prevalence = oof[oof["endpoint"] == selected_endpoint][["patient_id", "y_true"]].drop_duplicates()["y_true"].mean()
    for model_name in complete_models:
        data = pr_data[(pr_data["endpoint"] == selected_endpoint) & (pr_data["model_name"] == model_name)]
        if data.empty:
            continue
        label = f"{MODEL_LABEL[model_name]} ({data['pr_auc'].iloc[0]:.3f})"
        ax_c.plot(data["recall"], data["precision"], label=label, color=FAMILY_COLOR[MODEL_FAMILY[model_name]], linewidth=1.5, alpha=0.9)
    ax_c.axhline(prevalence, linestyle="--", color="#7B8791", linewidth=1, label=f"Prevalence ({prevalence:.2f})")
    ax_c.set_xlabel("Recall")
    ax_c.set_ylabel("Precision")
    ax_c.legend(frameon=False, loc="lower left", fontsize=8.5)
    panel_label(ax_c, "C")

    ax_d.set_title("Pooled ROC AUC by endpoint")
    heat_plot = pooled_heat.rename(index=MODEL_LABEL, columns=ENDPOINT_LABEL)
    sns.heatmap(
        heat_plot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.50,
        vmax=0.82,
        mask=heat_plot.isna(),
        cbar_kws={"label": "ROC AUC"},
        ax=ax_d,
    )
    for y, model in enumerate(heat_plot.index):
        for x, endpoint in enumerate(heat_plot.columns):
            if pd.isna(heat_plot.loc[model, endpoint]):
                ax_d.text(x + 0.5, y + 0.5, "NA", ha="center", va="center", color="#606970", fontweight="bold")
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    panel_label(ax_d, "D")

    if ed_variant:
        ax_e.set_title("Estimator sensitivity: mean change after pooling")
        rs = RESULTS / "source_data/reusability"
        legacy = pd.read_csv(rs / "legacy_patient_averaged/pooling_did.csv").set_index("endpoint")
        fold = pd.read_csv(rs / "foldaware_pooling_did.csv")
        fold = fold[fold["metric"] == "auc"].set_index("endpoint")
        families = [("foundation_mean_delta", "Foundation models", "#158F82", -0.2), ("tree_mean_delta", "Tree ensembles", "#CC6B4E", 0.2)]
        for i, endpoint in enumerate(ENDPOINT_ORDER):
            for column, _, colour, off in families:
                a, b = legacy.loc[endpoint, column], fold.loc[endpoint, column]
                ax_e.plot([i + off, i + off], [a, b], color=colour, linewidth=1.2, alpha=0.7)
                ax_e.scatter(i + off, a, s=46, facecolor="white", edgecolor=colour, linewidth=1.5, zorder=3)
                ax_e.scatter(i + off, b, s=46, color=colour, zorder=3)
        ax_e.axhline(0, color="#7B8791", linestyle="--", linewidth=1)
        ax_e.set_xticks(range(3))
        ax_e.set_xticklabels([ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER])
        ax_e.set_ylabel("Pooled minus cancer-specific\nwithin-cancer ROC AUC")
        ax_e.legend(
            handles=[Line2D([0], [0], marker="o", color="#158F82", linestyle="None", label="Foundation models", markersize=7),
                     Line2D([0], [0], marker="o", color="#CC6B4E", linestyle="None", label="Tree ensembles", markersize=7),
                     Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", linestyle="None", label="Predictions pooled across folds", markersize=7),
                     Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="Scored within each test set", markersize=7)],
            frameon=False, fontsize=8.0, loc="lower left",
        )
    else:
        # Paired within-cancer contrast; produced by within_cancer_pooling_contrast.py.
        ax_e.set_title("Effect of pooled training on within-cancer ROC AUC")
        contrast = pd.read_csv(SOURCE_ROOT / "figure_03_within_cancer_pooling_summary.csv")
        models_e = [m for m in MODEL_ORDER if m in set(contrast["model_name"])]
        offsets = {"os_3yr": -0.22, "os_5yr": 0.0, "extreme_os": 0.22}
        for y, model_name in enumerate(models_e):
            for endpoint in ENDPOINT_ORDER:
                row = contrast[(contrast["model_name"] == model_name) & (contrast["endpoint"] == endpoint)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ypos = y + offsets[endpoint]
                excludes_zero = row["mean_delta_ci_high"] < 0 or row["mean_delta_ci_low"] > 0
                ax_e.plot([row["mean_delta_ci_low"], row["mean_delta_ci_high"]], [ypos, ypos], color=ENDPOINT_COLOR[endpoint], linewidth=1.4)
                ax_e.scatter(
                    row["mean_delta"], ypos, color=ENDPOINT_COLOR[endpoint] if excludes_zero else "white",
                    edgecolor=ENDPOINT_COLOR[endpoint], linewidth=1.3, s=34, zorder=3,
                )
        ax_e.axvline(0, color="#7B8791", linestyle="--", linewidth=1)
        ax_e.set_yticks(range(len(models_e)))
        ax_e.set_yticklabels([MODEL_LABEL[m] for m in models_e])
        ax_e.invert_yaxis()
        ax_e.set_xlabel("Pooled minus cancer-specific ROC AUC (95% CI)")
        ax_e.legend(
            handles=[Line2D([0], [0], marker="o", color=ENDPOINT_COLOR[e], label=ENDPOINT_LABEL[e], markersize=6) for e in ENDPOINT_ORDER]
            + [Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white", linestyle="None", label="CI includes 0", markersize=6)],
            frameon=False, fontsize=8.5, loc="lower right",
        )
        ax_e.set_xlim(-0.145, 0.215)
    panel_label(ax_e, "E")

    ax_f.set_title("Task size and discrimination")
    for scope, marker, size in [("per_cancer", "o", 36), ("pooled", "s", 58)]:
        subset = task_size[task_size["scope"] == scope]
        for endpoint in ENDPOINT_ORDER:
            values = subset[subset["endpoint"] == endpoint]
            ax_f.scatter(values["n_total"], values["roc_auc"], color=ENDPOINT_COLOR[endpoint], marker=marker, s=size, alpha=0.8)
    ax_f.axhline(0.5, color="#7B8791", linestyle="--", linewidth=1)
    ax_f.set_xscale("log")
    ax_f.set_xlabel("Eligible patients (log scale)")
    ax_f.set_ylabel("Mean ROC AUC across evaluated models")
    handles = endpoint_handles + [
        Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="Per-cancer", markersize=7),
        Line2D([0], [0], marker="s", color="#555555", linestyle="None", label="Pooled", markersize=7),
    ]
    ax_f.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2)
    panel_label(ax_f, "F")

    save_figure(fig, "figure_03_pooling_effect_ed" if ed_variant else "figure_03_pooling_effect")


def build_figure_4(summary: pd.DataFrame) -> None:
    controls = pd.read_csv(CONTROL_ROOT / "cancer_shortcut_control_metrics.csv")
    separability = pd.read_csv(LANDSCAPE_ROOT / "cohort_separability.csv")
    heldout = pd.read_csv(STRESS_ROOT / "cancer_cohort_held_out_metrics.csv")
    permutation = pd.read_csv(STRESS_ROOT / "cancer_label_permutation_metrics.csv")
    controls.to_csv(SOURCE_ROOT / "figure_04_shortcut_controls.csv", index=False)
    separability.to_csv(SOURCE_ROOT / "figure_04_cohort_separability.csv", index=False)
    heldout.to_csv(SOURCE_ROOT / "figure_04_heldout_diagnostics.csv", index=False)
    permutation.to_csv(SOURCE_ROOT / "figure_04_permutation_distributions.csv", index=False)

    selected_controls = ["prevalence_only", "cancer_identity_only", "structural_zero_pattern", "linear_molecular"]
    control_label = {
        "prevalence_only": "Prevalence",
        "cancer_identity_only": "Cancer identity",
        "structural_zero_pattern": "Structural zero pattern",
        "linear_molecular": "Linear molecular",
    }
    control_color = {
        "prevalence_only": "#AAB2BB",
        "cancer_identity_only": "#B84A5A",
        "structural_zero_pattern": "#117A8B",
        "linear_molecular": "#426B8A",
    }

    pooled = summary[summary["scope"] == "pooled"][["endpoint", "model_name", "roc_auc_mean"]].copy()
    strongest = controls[controls["control"].isin(["cancer_identity_only", "structural_zero_pattern"])].groupby("endpoint")["roc_auc"].max()
    pooled["shortcut_reference_auc"] = pooled["endpoint"].map(strongest)
    pooled["auc_above_shortcut"] = pooled["roc_auc_mean"] - pooled["shortcut_reference_auc"]
    pooled.to_csv(SOURCE_ROOT / "figure_04_model_minus_shortcut.csv", index=False, na_rep="NA")

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.08])
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.suptitle("Cohort identity and structural missingness stress tests", fontweight="bold")

    for ax, metric, label, letter in [(ax_a, "roc_auc", "ROC AUC", "A"), (ax_b, "pr_auc", "PR AUC", "B")]:
        data = controls[controls["control"].isin(selected_controls)].copy()
        x = np.arange(len(ENDPOINT_ORDER))
        width = 0.19
        for idx, control in enumerate(selected_controls):
            values = data[data["control"] == control].set_index("endpoint").reindex(ENDPOINT_ORDER)[metric]
            ax.bar(x + (idx - 1.5) * width, values, width, label=control_label[control], color=control_color[control])
        ax.axhline(0.5, color="#7B8791", linestyle="--", linewidth=1)
        ax.set_xticks(x, [ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER])
        ax.set_ylabel(label)
        ax.set_ylim(0.35, 0.92 if metric == "roc_auc" else 0.95)
        ax.set_title(f"Shortcut controls: {label}")
        ax.legend(frameon=False, fontsize=8.4, ncol=2)
        panel_label(ax, letter)

    ax_c.set_title("Cancer-cohort identifiability")
    sep = separability.set_index("control").reindex(["molecular_values", "structural_zero_pattern"])
    labels = ["Molecular values", "Structural zero pattern"]
    x = np.arange(2)
    ax_c.bar(x - 0.18, sep["accuracy"], 0.36, label="Accuracy", color="#D08C3A")
    ax_c.bar(x + 0.18, sep["balanced_accuracy"], 0.36, label="Balanced accuracy", color="#7A6F80")
    ax_c.set_xticks(x, labels, rotation=12)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("Five-fold score")
    ax_c.legend(frameon=False, fontsize=8.5)
    for idx, value in enumerate(sep["balanced_accuracy"]):
        ax_c.text(idx + 0.18, value + 0.025, f"{value:.3f}", ha="center", fontweight="bold")
    panel_label(ax_c, "C")

    ax_d.set_title("Leave-one-cancer-out molecular diagnostic")
    held = heldout[(heldout["partition_type"] == "leave_one_cancer_out") & (heldout["control"] == "linear_molecular")]
    held_heat = held.pivot(index="partition_value", columns="endpoint", values="roc_auc").reindex(index=CANCER_ORDER, columns=ENDPOINT_ORDER)
    held_heat = held_heat.rename(columns=ENDPOINT_LABEL)
    sns.heatmap(held_heat, annot=True, fmt=".2f", cmap="vlag", center=0.5, vmin=0.35, vmax=0.65, cbar_kws={"label": "ROC AUC"}, ax=ax_d)
    ax_d.set_xlabel("")
    ax_d.set_ylabel("Held-out cancer")
    panel_label(ax_d, "D")

    ax_e.set_title("Label-permutation null distributions")
    sns.violinplot(data=permutation, x="permutation", y="roc_auc", hue="endpoint", hue_order=ENDPOINT_ORDER, palette=ENDPOINT_COLOR, cut=0, inner="quart", ax=ax_e)
    ax_e.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax_e.set_xticks([0, 1], ["Global", "Within cancer"])
    ax_e.set_xlabel("")
    ax_e.set_ylabel("ROC AUC")
    handles, labels = ax_e.get_legend_handles_labels()
    ax_e.legend(handles, [ENDPOINT_LABEL[e] for e in ENDPOINT_ORDER], frameon=False, fontsize=8.5)
    panel_label(ax_e, "E")

    ax_f.set_title("Molecular models relative to shortcut baseline")
    delta_heat = pooled.pivot(index="model_name", columns="endpoint", values="auc_above_shortcut").reindex(MODEL_ORDER)[ENDPOINT_ORDER]
    delta_plot = delta_heat.rename(index=MODEL_LABEL, columns=ENDPOINT_LABEL)
    sns.heatmap(
        delta_plot,
        annot=True,
        fmt="+.2f",
        cmap="vlag",
        center=0,
        vmin=-0.25,
        vmax=0.15,
        mask=delta_plot.isna(),
        cbar_kws={"label": "Delta ROC AUC"},
        ax=ax_f,
    )
    for y, model in enumerate(delta_plot.index):
        for x, endpoint in enumerate(delta_plot.columns):
            if pd.isna(delta_plot.loc[model, endpoint]):
                ax_f.text(x + 0.5, y + 0.5, "NA", ha="center", va="center", color="#606970", fontweight="bold")
    ax_f.set_xlabel("")
    ax_f.set_ylabel("")
    panel_label(ax_f, "F")

    save_figure(fig, "figure_04_shortcut_stress_tests")


def feature_modality_stability() -> pd.DataFrame:
    rows = []
    for path in FOLD_ROOT.glob("pooled/*/ALL/repeat_*_fold_*/selected_features.csv"):
        endpoint = path.parts[-4]
        fold_name = path.parent.name
        selected = pd.read_csv(path)
        counts = selected["modality"].value_counts()
        for modality, count in counts.items():
            rows.append({"endpoint": endpoint, "fold": fold_name, "modality": modality, "count": count})
    return pd.DataFrame(rows)


def build_figure_5(metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    selected_models = ["random_forest", "catboost", "autogluon", "tabfm_default", "tabpfn_v3"]
    oof = load_oof_predictions(metrics, selected_models, ENDPOINT_ORDER)
    calibration_rows = []
    for (endpoint, model_name), group in oof.groupby(["endpoint", "model_name"]):
        observed, predicted = calibration_curve(group["y_true"], group["probability"], n_bins=8, strategy="quantile")
        calibration_rows.extend(
            {"endpoint": endpoint, "model_name": model_name, "mean_predicted": x, "observed": y}
            for x, y in zip(predicted, observed)
        )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(SOURCE_ROOT / "figure_05_calibration_curves.csv", index=False)

    pooled = summary[summary["scope"] == "pooled"].copy()
    pooled.to_csv(SOURCE_ROOT / "figure_05_pooled_probabilistic_metrics.csv", index=False, na_rep="NA")
    modalities = feature_modality_stability()
    modalities.to_csv(SOURCE_ROOT / "figure_05_feature_modalities_across_folds.csv", index=False)

    runtime = (
        metrics[metrics["roc_auc"].notna()]
        .groupby(["model_name", "model", "family"])
        .agg(fit_time_s=("fit_time_s", "median"), predict_time_s=("predict_time_s", "median"))
        .reset_index()
    )
    runtime.to_csv(SOURCE_ROOT / "figure_05_runtime.csv", index=False)

    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.06, 1.03, 1.06])
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    fig.suptitle("Probabilistic performance and operational tradeoffs", fontweight="bold")

    ax_a.set_title("Pooled 5-year calibration")
    data_a = calibration[calibration["endpoint"] == "os_5yr"]
    for model_name in selected_models:
        data = data_a[data_a["model_name"] == model_name]
        ax_a.plot(data["mean_predicted"], data["observed"], marker="o", markersize=3, linewidth=1.2, color=FAMILY_COLOR[MODEL_FAMILY[model_name]], label=MODEL_LABEL[model_name])
    ax_a.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1)
    ax_a.set_xlabel("Mean predicted probability")
    ax_a.set_ylabel("Observed event fraction")
    ax_a.legend(frameon=False, fontsize=8.5)
    panel_label(ax_a, "A")

    for ax, metric, title, letter, vmin, vmax in [
        (ax_b, "brier_mean", "Pooled Brier score", "B", 0.14, 0.30),
        (ax_c, "log_loss_mean", "Pooled log loss", "C", 0.45, 0.95),
    ]:
        heat = pooled.pivot(index="model_name", columns="endpoint", values=metric).reindex(MODEL_ORDER)[ENDPOINT_ORDER]
        heat_plot = heat.rename(index=MODEL_LABEL, columns=ENDPOINT_LABEL)
        sns.heatmap(heat_plot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=vmin, vmax=vmax, mask=heat_plot.isna(), cbar_kws={"label": title}, ax=ax)
        for y, model in enumerate(heat_plot.index):
            for x, endpoint in enumerate(heat_plot.columns):
                if pd.isna(heat_plot.loc[model, endpoint]):
                    ax.text(x + 0.5, y + 0.5, "NA", ha="center", va="center", color="#606970", fontweight="bold")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        panel_label(ax, letter)

    ax_d.set_title("Validation-threshold operating characteristics")
    for endpoint in ENDPOINT_ORDER:
        data = pooled[pooled["endpoint"] == endpoint]
        ax_d.scatter(data["specificity_mean"], data["sensitivity_mean"], color=ENDPOINT_COLOR[endpoint], label=ENDPOINT_LABEL[endpoint], s=42, alpha=0.85)
    ax_d.plot([0, 1], [1, 0], linestyle=":", color="#AAB2BB")
    ax_d.set_xlim(0.3, 0.9)
    ax_d.set_ylim(0.3, 0.95)
    ax_d.set_xlabel("Specificity")
    ax_d.set_ylabel("Sensitivity")
    ax_d.legend(frameon=False, fontsize=8.5)
    panel_label(ax_d, "D")

    ax_e.set_title("Selected modality composition across pooled folds")
    mod = modalities.groupby(["endpoint", "modality"])["count"].mean().unstack(fill_value=0).reindex(ENDPOINT_ORDER)
    mod = mod.div(mod.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(mod))
    palette = sns.color_palette("Set2", n_colors=max(1, len(mod.columns)))
    for color, modality in zip(palette, mod.columns):
        ax_e.bar(np.arange(len(mod)), mod[modality], bottom=bottom, label=modality, color=color)
        bottom += mod[modality].to_numpy()
    ax_e.set_xticks(np.arange(len(mod)), [ENDPOINT_LABEL[e] for e in mod.index])
    ax_e.set_ylabel("Mean share of 100 selected features (%)")
    ax_e.legend(frameon=False, fontsize=8.5, ncol=2)
    panel_label(ax_e, "E")

    ax_f.set_title("Median computational time per fold")
    run = runtime.set_index("model_name").reindex(MODEL_ORDER).dropna(subset=["fit_time_s", "predict_time_s"])
    ypos = np.arange(len(run))
    ax_f.barh(ypos, run["fit_time_s"], color="#D08C3A", label="Fit")
    ax_f.barh(ypos, run["predict_time_s"], left=run["fit_time_s"], color="#426B8A", label="Predict")
    ax_f.set_yticks(ypos, [MODEL_LABEL[m] for m in run.index])
    ax_f.set_xscale("symlog", linthresh=0.01)
    ax_f.set_xlabel("Seconds per fold (symlog scale)")
    ax_f.legend(frameon=False, fontsize=8.5)
    panel_label(ax_f, "F")

    save_figure(fig, "figure_05_probabilistic_and_compute")


def format_value(value: float | int, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_tables(summary: pd.DataFrame, manifest: pd.DataFrame) -> None:
    cohort_counts = pd.read_csv(LANDSCAPE_ROOT / "cohort_counts.csv")
    first = manifest.sort_values(["repeat", "fold"]).drop_duplicates(["scope", "endpoint", "cancer"])
    extreme = first[(first["scope"] == "per_cancer") & (first["endpoint"] == "extreme_os")][["cancer", "n_total"]].rename(columns={"n_total": "n_extreme"})
    table1 = cohort_counts.merge(extreme, on="cancer", how="left").set_index("cancer").reindex(CANCER_ORDER).reset_index()
    table1.to_csv(TABLE_ROOT / "table_01_cohort_characteristics.csv", index=False, na_rep="NA")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Cohort & Raw & 3-year eligible & 5-year eligible & Extreme eligible \\\\",
        "\\midrule",
    ]
    for row in table1.itertuples():
        lines.append(f"{row.cancer} & {int(row.n_raw)} & {int(row.n_os_3yr)} & {format_value(row.n_os_5yr, 0)} & {format_value(row.n_extreme, 0)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLE_ROOT / "table_01_cohort_characteristics.tex").write_text("\n".join(lines) + "\n")

    auc = summary.pivot(index="model_name", columns=["scope", "endpoint"], values="roc_auc_mean")
    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in auc.index:
            continue
        row = {"model_name": model_name, "model": MODEL_LABEL[model_name], "family": MODEL_FAMILY[model_name]}
        for scope in ["per_cancer", "pooled"]:
            for endpoint in ENDPOINT_ORDER:
                row[f"{scope}_{endpoint}"] = auc.loc[model_name].get((scope, endpoint), np.nan)
        rows.append(row)
    table2 = pd.DataFrame(rows)
    table2["within_macro"] = table2[[f"per_cancer_{e}" for e in ENDPOINT_ORDER]].mean(axis=1)
    table2.to_csv(TABLE_ROOT / "table_02_model_auc_summary.csv", index=False, na_rep="NA")
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Family & Within-cancer & Pooled 3-year & Pooled 5-year & Pooled extreme \\\\",
        "\\midrule",
    ]
    for row in table2.itertuples():
        lines.append(
            f"{row.model} & {row.family} & {format_value(row.within_macro)} & {format_value(row.pooled_os_3yr)} & {format_value(row.pooled_os_5yr)} & {format_value(row.pooled_extreme_os)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLE_ROOT / "table_02_model_auc_summary.tex").write_text("\n".join(lines) + "\n")

    controls = pd.read_csv(CONTROL_ROOT / "cancer_shortcut_control_metrics.csv")
    heldout = pd.read_csv(STRESS_ROOT / "cancer_cohort_held_out_metrics.csv")
    held_means = heldout[
        (heldout["partition_type"] == "leave_one_cancer_out") & (heldout["control"] == "linear_molecular")
    ].groupby("endpoint")["roc_auc"].mean()
    table3_rows = []
    for endpoint in ENDPOINT_ORDER:
        control = controls[controls["endpoint"] == endpoint].set_index("control")
        pooled_best = summary[(summary["scope"] == "pooled") & (summary["endpoint"] == endpoint)]["roc_auc_mean"].max()
        table3_rows.append(
            {
                "endpoint": endpoint,
                "cancer_identity_auc": control.loc["cancer_identity_only", "roc_auc"],
                "structural_zero_auc": control.loc["structural_zero_pattern", "roc_auc"],
                "best_evaluated_model_auc": pooled_best,
                "heldout_linear_auc": held_means.get(endpoint, np.nan),
            }
        )
    table3 = pd.DataFrame(table3_rows)
    table3.to_csv(TABLE_ROOT / "table_03_shortcut_summary.csv", index=False)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Endpoint & Cancer identity & Structural zero & Best evaluated model & Held-out molecular \\\\",
        "\\midrule",
    ]
    for row in table3.itertuples():
        lines.append(
            f"{ENDPOINT_LABEL[row.endpoint]} & {row.cancer_identity_auc:.3f} & {row.structural_zero_auc:.3f} & {row.best_evaluated_model_auc:.3f} & {row.heldout_linear_auc:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLE_ROOT / "table_03_shortcut_summary.tex").write_text("\n".join(lines) + "\n")


def write_manifest() -> None:
    rows = []
    figure_sources = {
        "Figure 1": ["figure_01_cohort_flow.csv", "figure_01_class_counts.csv", "figure_01_modality_composition.csv"],
        "Figure 2": ["figure_02_within_cancer_model_summary.csv", "figure_02_within_cancer_auc_heatmap.csv", "figure_02_paired_auc_delta.csv", "figure_02_pr_auc_lift.csv", "figure_02_runtime_pareto.csv"],
        "Figure 3": ["figure_03_patient_oof_predictions.csv", "figure_03_roc_curves.csv", "figure_03_pr_curves.csv", "figure_03_pooling_shift.csv", "figure_03_pooled_auc_heatmap.csv", "figure_03_pooled_subgroup_auc.csv", "figure_03_within_cancer_pooling_contrast.csv", "figure_03_within_cancer_pooling_summary.csv", "figure_03_task_size_auc.csv"],
        "Figure 4": ["figure_04_shortcut_controls.csv", "figure_04_cohort_separability.csv", "figure_04_heldout_diagnostics.csv", "figure_04_permutation_distributions.csv", "figure_04_model_minus_shortcut.csv"],
        "Figure 5": ["figure_05_calibration_curves.csv", "figure_05_pooled_probabilistic_metrics.csv", "figure_05_feature_modalities_across_folds.csv", "figure_05_runtime.csv"],
    }
    for figure, sources in figure_sources.items():
        for source in sources:
            rows.append({"figure": figure, "source_table": f"paper/figures/source_data/{source}", "generator": "paper/analysis/build_manuscript_assets.py"})
    pd.DataFrame(rows).to_csv(SOURCE_ROOT / "figure_source_manifest.csv", index=False)


def main() -> None:
    setup()
    metrics, summary = load_metrics()
    manifest = load_fold_manifest()
    build_figure_1(manifest)
    build_figure_2(metrics, manifest)
    build_figure_3(metrics, summary, manifest)
    build_figure_3(metrics, summary, manifest, ed_variant=True)
    build_figure_4(summary)
    build_figure_5(metrics, summary)
    build_tables(summary, manifest)
    write_manifest()
    print(json.dumps({"figures": 5, "source_tables": len(list(SOURCE_ROOT.glob("*.csv"))), "generated_tables": len(list(TABLE_ROOT.glob("*")))}, indent=2))


if __name__ == "__main__":
    main()
