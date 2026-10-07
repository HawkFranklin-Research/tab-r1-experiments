from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


class Aggregator:
    def __init__(self, runs_root: str, results_dir: str) -> None:
        self.runs_root = Path(runs_root)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ev_tabpfn.aggregator")

    def collect_results(self) -> pd.DataFrame:
        all_data = []
        if not self.runs_root.exists():
            return pd.DataFrame()
        for dataset_dir in self.runs_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for run_path in dataset_dir.iterdir():
                metrics_json = run_path / "metrics" / "metrics_summary.json"
                if not metrics_json.exists():
                    continue
                data = json.loads(metrics_json.read_text())
                for row in data.get("rows", []):
                    row["dataset"] = dataset_dir.name
                    all_data.append(row)
        return pd.DataFrame(all_data)

    def _format_table(self, df: pd.DataFrame, metrics: list[str]) -> str:
        agg_dict = {m: ["mean", "std"] for m in metrics if m in df.columns}
        agg_dict["fit_time_s"] = ["mean"]
        summary = df[df["status"] == "success"].groupby("model_name").agg(agg_dict)
        metric_names = [m for m in metrics if m in df.columns]
        header = "  ┌────────────" + "┬───────────────" * len(metric_names) + "┬───────────────────┐"
        col_names = "  │ Model      " + "".join(f"│ {m.upper():<14} " for m in metric_names) + "│ Mean Fit Time (s) │"
        divider = "  ├────────────" + "┼───────────────" * len(metric_names) + "┼───────────────────┤"
        table_lines = [header, col_names, divider]
        for model, row in summary.iterrows():
            line = f"  │ {model:<10} "
            for m in metric_names:
                mean = row[(m, "mean")]
                std = row[(m, "std")]
                val_str = f"{mean:.3f} +/- {std:.2f}" if not np.isnan(std) else f"{mean:.3f} +/- 0.00"
                line += f"│ {val_str:<14} "
            line += f"│ {row[('fit_time_s', 'mean')]:<17.3f}s │"
            table_lines.append(line)
        footer = "  └────────────" + "┴───────────────" * len(metric_names) + "┴───────────────────┘"
        table_lines.append(footer)
        return "\n".join(table_lines)

    def run_aggregation(self) -> dict[str, str]:
        df = self.collect_results()
        outputs: dict[str, str] = {}
        if df.empty:
            return outputs

        cls_df = df[df["task_type"].isin(["binary", "multiclass"])]
        if not cls_df.empty:
            table = self._format_table(cls_df, ["roc_auc", "accuracy"])
            path = self.results_dir / "aggregate_classification.md"
            path.write_text("# Aggregate Classification Performance\n\n" + table + "\n")
            summary_path = self.results_dir / "benchmark_summary.md"
            summary_path.write_text(
                "# Final Benchmark Evaluation Summary\n\n"
                "This table summarizes all successfully evaluated datasets in this batch.\n\n"
                + table
                + "\n\n*Note: Mean +/- Std Dev computed across all successful runs.*\n"
            )
            outputs["aggregate_classification"] = str(path)
            outputs["benchmark_summary"] = str(summary_path)
            try:
                from ev_tabpfn.reporting.comprehensive_plots import ComprehensiveVisualizer

                visualizer = ComprehensiveVisualizer(str(self.runs_root), str(self.results_dir))
                for key, value in {
                    "benchmark_roc_grid": visualizer.plot_roc_grid(),
                    "benchmark_mean_roc_envelope": visualizer.plot_mean_roc_envelope(),
                    "benchmark_score_distribution": visualizer.plot_performance_boxplot(cls_df),
                }.items():
                    if value is not None:
                        outputs[key] = value
            except Exception as exc:
                self.logger.error("Failed to generate comprehensive plots: %s", exc)

        reg_df = df[df["task_type"] == "regression"]
        if not reg_df.empty:
            table = self._format_table(reg_df, ["r2", "rmse"])
            path = self.results_dir / "aggregate_regression.md"
            path.write_text("# Aggregate Regression Performance\n\n" + table + "\n")
            outputs["aggregate_regression"] = str(path)

        return outputs
