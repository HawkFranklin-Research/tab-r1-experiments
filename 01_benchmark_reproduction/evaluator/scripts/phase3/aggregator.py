import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

class Aggregator:
    """Aggregates results across multiple runs to compute Mean ± Std/CI for the final report."""
    
    def __init__(self, runs_root: str, results_dir: str):
        self.runs_root = runs_root
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger = logging.getLogger("Aggregator")
        logging.basicConfig(level=logging.INFO)

    def collect_results(self) -> pd.DataFrame:
        """Crawl the runs directory and collect all metrics."""
        all_data = []
        
        for dataset_dir in os.listdir(self.runs_root):
            dataset_path = os.path.join(self.runs_root, dataset_dir)
            if not os.path.isdir(dataset_path): continue
            
            for run_timestamp in os.listdir(dataset_path):
                run_path = os.path.join(dataset_path, run_timestamp)
                metrics_json = os.path.join(run_path, "metrics/metrics_summary.json")
                
                if os.path.exists(metrics_json):
                    with open(metrics_json, 'r') as f:
                        data = json.load(f)
                        for row in data.get('rows', []):
                            row['dataset'] = dataset_dir
                            all_data.append(row)
        
        return pd.DataFrame(all_data)

    def _format_table(self, df: pd.DataFrame, metrics: List[str], task_label: str) -> str:
        """Formats an aggregated table with box-drawing characters."""
        
        # Calculate Mean and Std Dev grouped by model
        agg_dict = {m: ['mean', 'std'] for m in metrics}
        agg_dict['fit_time_s'] = ['mean']
        
        summary = df[df['status'] == 'success'].groupby('model_name').agg(agg_dict)
        
        # Build the table
        header = "  ┌────────────"
        for m in metrics: header += "┬───────────────"
        header += "┬───────────────────┐"
        
        col_names = "  │ Model      "
        for m in metrics: col_names += f"│ {m.upper():<14} "
        col_names += "│ Mean Fit Time (s) │"
        
        divider = "  ├────────────"
        for m in metrics: divider += "┼───────────────"
        divider += "┼───────────────────┤"
        
        table_lines = [header, col_names, divider]
        
        for model, row in summary.iterrows():
            line = f"  │ {model:<10} "
            for m in metrics:
                mean = row[(m, 'mean')]
                std = row[(m, 'std')]
                val_str = f"{mean:.3f} ± {std:.2f}" if not np.isnan(std) else f"{mean:.3f} ± 0.00"
                line += f"│ {val_str:<14} "
            
            fit_time = row[('fit_time_s', 'mean')]
            line += f"│ {fit_time:<17.3f}s │"
            table_lines.append(line)
            
        footer = "  └────────────"
        for m in metrics: footer += "┴───────────────"
        footer += "┴───────────────────┘"
        table_lines.append(footer)
        
        return "\n".join(table_lines)

    def run_aggregation(self):
        self.logger.info("Collecting results for aggregation...")
        df = self.collect_results()
        
        if df.empty:
            self.logger.error("No results found to aggregate.")
            return

        # 1. Classification Summary
        cls_df = df[df['task_type'].isin(['binary', 'multiclass'])]
        if not cls_df.empty:
            cls_table = self._format_table(cls_df, ['roc_auc', 'accuracy'], "Classification")
            
            # Save standard aggregate
            with open(os.path.join(self.results_dir, "aggregate_classification.md"), 'w') as f:
                f.write("# Aggregate Classification Performance\n\n")
                f.write(cls_table)
            
            # Save formal final summary for the benchmark
            summary_path = os.path.join(self.results_dir, "benchmark_summary.md")
            with open(summary_path, 'w') as f:
                f.write("# Final Benchmark Evaluation Summary\n\n")
                f.write("This table summarizes the performance across all successfully evaluated datasets in this batch.\n\n")
                f.write(cls_table)
                f.write("\n\n*Note: Mean ± Std Dev computed across all successful runs.*\n")
            
            self.logger.info("Generic benchmark summary table generated at %s", summary_path)
            
            # Generate Comprehensive Plots
            try:
                from scripts.phase3.comprehensive_plots import ComprehensiveVisualizer
                cv = ComprehensiveVisualizer(self.runs_root, self.results_dir)
                self.logger.info("Generating comprehensive benchmark plots...")
                cv.plot_roc_grid()
                cv.plot_mean_roc_envelope()
                cv.plot_performance_boxplot(cls_df)
                cv.plot_radar_auc(cls_df)
                self.logger.info("Comprehensive plots saved to %s", self.results_dir)
            except Exception as e:
                self.logger.error("Failed to generate comprehensive plots: %s", str(e))

        # 2. Regression Summary
        reg_df = df[df['task_type'] == 'regression']
        if not reg_df.empty:
            reg_table = self._format_table(reg_df, ['r2', 'rmse'], "Regression")
            with open(os.path.join(self.results_dir, "aggregate_regression.md"), 'w') as f:
                f.write("# Aggregate Regression Performance\n\n")
                f.write(reg_table)
            self.logger.info("Regression aggregate table generated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 Aggregator")
    parser.add_argument("--runs_root", type=str, default="/home/prime/Documents/g3/tab-r1/results-s2/runs")
    parser.add_argument("--results_dir", type=str, default="/home/prime/Documents/g3/tab-r1/results-s2/results")
    args = parser.parse_args()
    
    agg = Aggregator(args.runs_root, args.results_dir)
    agg.run_aggregation()
