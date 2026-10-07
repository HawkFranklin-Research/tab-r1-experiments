from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None


MODEL_COLORS = {
    "tabpfn": "#1f77b4",
    "catboost": "#d62728",
    "autogluon": "#9467bd",
    "xgboost": "#ff7f0e",
    "lightgbm": "#2ca02c",
    "random_forest": "#8c564b",
    "logistic_regression": "#7f7f7f",
    "ridge": "#7f7f7f",
}


class ComprehensiveVisualizer:
    def __init__(self, runs_root: str, output_dir: str) -> None:
        if plt is None or sns is None:
            raise RuntimeError("matplotlib and seaborn are required for comprehensive plots.")
        self.runs_root = Path(runs_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        self.logger = logging.getLogger("ev_tabpfn.comprehensive_plots")

    def _load_all_predictions(self) -> list[dict]:
        all_preds = []
        if not self.runs_root.exists():
            return all_preds
        for dataset_dir in self.runs_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for run_path in dataset_dir.iterdir():
                pred_dir = run_path / "predictions"
                if not pred_dir.exists():
                    continue
                for pred_file in pred_dir.glob("*_predictions.csv"):
                    all_preds.append(
                        {
                            "dataset": dataset_dir.name,
                            "model": pred_file.name.replace("_predictions.csv", ""),
                            "df": pd.read_csv(pred_file),
                        }
                    )
        return all_preds

    def plot_roc_grid(self) -> str | None:
        from sklearn.metrics import auc, roc_curve

        preds_list = self._load_all_predictions()
        datasets = sorted({p["dataset"] for p in preds_list})
        if not datasets:
            return None

        cols = 3
        rows = (len(datasets) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), squeeze=False)

        for i, ds_name in enumerate(datasets):
            ax = axes[i // cols, i % cols]
            ds_preds = [p for p in preds_list if p["dataset"] == ds_name]
            has_plots = False
            for p in ds_preds:
                prob_cols = [c for c in p["df"].columns if c.startswith("prob_")]
                if len(prob_cols) != 2 or "y_true_encoded" not in p["df"].columns:
                    continue
                try:
                    prob_col = prob_cols[-1]
                    fpr, tpr, _ = roc_curve(p["df"]["y_true_encoded"], p["df"][prob_col])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f"{p['model'].title()} ({roc_auc:.3f})")
                    has_plots = True
                except ValueError as exc:
                    self.logger.error("Skipping ROC for %s on %s: %s", p["model"], ds_name, exc)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_title(f"Dataset: {ds_name.replace('_dataset', '').title()}")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            if has_plots:
                ax.legend(fontsize="small", loc="lower right")

        for j in range(len(datasets), rows * cols):
            axes[j // cols, j % cols].axis("off")

        plt.tight_layout()
        path = self.output_dir / "benchmark_roc_grid.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_mean_roc_envelope(self) -> str | None:
        from sklearn.metrics import roc_curve

        preds_list = self._load_all_predictions()
        models = sorted({p["model"] for p in preds_list})
        mean_fpr = np.linspace(0, 1, 100)
        plot_data = []
        for model in models:
            for p in [item for item in preds_list if item["model"] == model]:
                prob_cols = [c for c in p["df"].columns if c.startswith("prob_")]
                if len(prob_cols) != 2 or "y_true_encoded" not in p["df"].columns:
                    continue
                try:
                    fpr, tpr, _ = roc_curve(p["df"]["y_true_encoded"], p["df"][prob_cols[-1]])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    for fpr_value, tpr_value in zip(mean_fpr, interp_tpr):
                        plot_data.append({"Model": model.title(), "FPR": fpr_value, "TPR": tpr_value})
                except ValueError:
                    continue

        if not plot_data:
            return None
        df_plot = pd.DataFrame(plot_data)
        palette = {
            label: MODEL_COLORS.get(label.split(" (")[0].lower(), "#333333")
            for label in df_plot["Model"].unique()
        }
        plt.figure(figsize=(10, 8))
        sns.lineplot(data=df_plot, x="FPR", y="TPR", hue="Model", palette=palette, errorbar="sd", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.title("Benchmark-wide Mean ROC with Std Dev Envelope", fontsize=16)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        path = self.output_dir / "benchmark_mean_roc_envelope.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_performance_boxplot(self, summary_df: pd.DataFrame) -> str | None:
        if summary_df.empty:
            return None
        plot_df = summary_df.melt(
            id_vars=["model_name"],
            value_vars=[col for col in ["roc_auc", "accuracy"] if col in summary_df.columns],
            var_name="Metric",
            value_name="Score",
        )
        if plot_df.empty:
            return None
        plt.figure(figsize=(12, 7))
        sns.catplot(data=plot_df, x="model_name", y="Score", hue="Metric", kind="box", height=6, aspect=1.5)
        sns.stripplot(data=plot_df, x="model_name", y="Score", hue="Metric", dodge=True, alpha=0.5, palette="dark:black", legend=False)
        plt.title("Score Distribution across Datasets", fontsize=16)
        plt.ylim([0.5, 1.05])
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / "benchmark_score_distribution.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

