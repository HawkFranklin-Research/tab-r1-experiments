from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None


METRIC_COLUMNS = [
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "log_loss",
]


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    subset = df[[col for col in columns if col in df.columns]].copy()
    headers = list(subset.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _plot_metric_means(df: pd.DataFrame, output_dir: Path) -> list[str]:
    if plt is None or sns is None or df.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = df[df["status"] == "success"].copy() if "status" in df.columns else df.copy()
    available = [metric for metric in METRIC_COLUMNS if metric in successful.columns]
    if not available:
        return []

    saved: list[str] = []
    long_df = successful.melt(id_vars=["model_name"], value_vars=available, var_name="metric", value_name="value").dropna()
    if not long_df.empty:
        sns.set_theme(style="whitegrid", context="talk")
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=long_df, x="metric", y="value", hue="model_name", ax=ax, palette="mako")
        ax.set_title("Mean metrics by model")
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        path = output_dir / "mean_metrics_by_model.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved.append(str(path))

    for metric in ["roc_auc", "pr_auc", "f1", "log_loss"]:
        if metric not in successful.columns:
            continue
        pivot = successful.pivot_table(index="dataset", columns="model_name", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig_height = max(5, 0.35 * len(pivot))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", linewidths=0.4, ax=ax)
        ax.set_title(f"{metric.upper()} by dataset and model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Dataset")
        fig.tight_layout()
        path = output_dir / f"{metric}_dataset_model_heatmap.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        saved.append(str(path))
    return saved


def _mean_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    successful = df[df["status"] == "success"].copy()
    available = [metric for metric in METRIC_COLUMNS if metric in successful.columns]
    if not available:
        return pd.DataFrame(), pd.DataFrame()
    by_model = successful.groupby("model_name", as_index=False)[available].mean()
    by_endpoint = pd.DataFrame()
    if "endpoint" in successful.columns:
        by_endpoint = successful.groupby(["endpoint", "model_name"], as_index=False)[available].mean()
    return by_model, by_endpoint


def build_report(output_root: Path, report_path: Path, title: str) -> dict[str, Any]:
    aggregate_dir = output_root / "aggregate"
    metrics = _read_table(aggregate_dir / "all_model_metrics.csv")
    subgroup = _read_table(aggregate_dir / "subgroup_metrics_by_cancer.csv")
    run_config_path = aggregate_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    report_path.parent.mkdir(parents=True, exist_ok=True)

    by_model, by_endpoint = _mean_tables(metrics)
    plots = _plot_metric_means(metrics, report_path.parent / "plots")

    split_columns = [
        "dataset",
        "endpoint",
        "source_cancers",
        "n_total",
        "n_train",
        "n_val",
        "n_test",
        "class_0_train",
        "class_1_train",
        "class_0_test",
        "class_1_test",
    ]
    metric_columns = [
        "dataset",
        "endpoint",
        "model_name",
        "status",
        "accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "log_loss",
    ]
    subgroup_columns = [
        "dataset",
        "model_name",
        "cancer_type",
        "n_test",
        "class_0_test",
        "class_1_test",
        "pr_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
    ]

    unique_splits = metrics.drop_duplicates("dataset") if "dataset" in metrics.columns else pd.DataFrame()
    lines = [
        f"# {title}",
        "",
        "This report is generated from saved prediction and metric artifacts. It does not rerun models.",
        "",
        "## Run Configuration",
        "",
        "```json",
        json.dumps(run_config, indent=2, default=str),
        "```",
        "",
        "## Dataset and Split Counts",
        "",
        _markdown_table(unique_splits, split_columns),
        "",
        "## Mean Metrics by Model",
        "",
        _markdown_table(by_model, ["model_name", *METRIC_COLUMNS]),
        "",
        "## Mean Metrics by Endpoint and Model",
        "",
        _markdown_table(by_endpoint, ["endpoint", "model_name", *METRIC_COLUMNS]),
        "",
        "## Dataset-Level Model Metrics",
        "",
        _markdown_table(metrics, metric_columns),
        "",
        "## Cancer Subgroup Metrics",
        "",
        _markdown_table(subgroup, subgroup_columns),
        "",
        "## Generated Plots",
        "",
    ]
    if plots:
        lines.extend([f"- `{path}`" for path in plots])
    else:
        lines.append("_No plots generated. Install matplotlib and seaborn if plotting is unavailable._")
    lines.append("")
    report_path.write_text("\n".join(lines))

    if not by_model.empty:
        by_model.to_csv(report_path.parent / "mean_metrics_by_model.csv", index=False)
    if not by_endpoint.empty:
        by_endpoint.to_csv(report_path.parent / "mean_metrics_by_endpoint_model.csv", index=False)
    return {"report_path": str(report_path), "plots": plots, "metric_rows": int(len(metrics)), "subgroup_rows": int(len(subgroup))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Markdown and plot reports from cancer OS experiment outputs.")
    parser.add_argument("--output-root", required=True, help="Model output root containing aggregate/all_model_metrics.csv.")
    parser.add_argument("--report-path", required=True, help="Markdown report destination.")
    parser.add_argument("--title", default="Cancer OS Foundation Model Experiment")
    args = parser.parse_args()

    result = build_report(
        output_root=Path(args.output_root).expanduser().resolve(),
        report_path=Path(args.report_path).expanduser().resolve(),
        title=args.title,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
