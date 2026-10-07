import pandas as pd
import json
import os
import logging
from typing import List, Dict

class Reporter:
    """Reporting engine for Evaluate-TABPFN to generate publication-grade Markdown tables."""
    
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(run_dir))), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load metadata and metrics
        with open(os.path.join(run_dir, "metadata/dataset_metadata.json"), 'r') as f:
            self.dataset_metadata = json.load(f)
        
        with open(os.path.join(run_dir, "metrics/metrics_summary.json"), 'r') as f:
            self.metrics_summary = json.load(f)
            
        self.task_type = self.dataset_metadata['task_type']
        self.dataset_name = self.dataset_metadata['dataset_name']
        
        self.logger = logging.getLogger("Reporter")
        logging.basicConfig(level=logging.INFO)

    def _format_value(self, val, std=None):
        if val is None:
            return "N/A"
        if std is not None:
            return f"{val:.4f} ± {std:.4f}"
        return f"{val:.4f}"

    def generate_markdown_table(self) -> str:
        """Generates a Markdown table for the current run."""
        rows = self.metrics_summary.get('rows', [])
        
        if self.task_type in ["binary", "multiclass"]:
            headers = ["Model", "ROC AUC", "Accuracy", "F1", "Fit Time (s)"]
            table = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
            
            for row in rows:
                if row['status'] != 'success':
                    table.append(f"| {row['model_name']} | FAILED | {row.get('error_type', 'Error')} | - | - |")
                    continue
                
                table.append(f"| {row['model_name']} | {self._format_value(row.get('roc_auc'))} | {self._format_value(row.get('accuracy'))} | {self._format_value(row.get('f1'))} | {row['fit_time_s']:.3f}s |")
                
        elif self.task_type == "regression":
            headers = ["Model", "R2", "RMSE", "MAE", "Fit Time (s)"]
            table = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
            
            for row in rows:
                if row['status'] != 'success':
                    table.append(f"| {row['model_name']} | FAILED | {row.get('error_type', 'Error')} | - | - |")
                    continue
                
                table.append(f"| {row['model_name']} | {self._format_value(row.get('r2'))} | {self._format_value(row.get('rmse'))} | {self._format_value(row.get('mae'))} | {row['fit_time_s']:.3f}s |")
        
        return "\n".join(table)

    def save_report(self):
        """Saves the result to the appropriate category file in the results/ directory."""
        filename_map = {
            "binary": "binary_classification.md",
            "multiclass": "multiclass_classification.md",
            "regression": "regression_single.md" # Assuming single for now
        }
        
        filename = filename_map.get(self.task_type, "other_results.md")
        report_path = os.path.join(self.results_dir, filename)
        
        md_table = self.generate_markdown_table()
        
        with open(report_path, 'a') as f:
            f.write(f"\n## Dataset: {self.dataset_name}\n")
            f.write(f"Task Type: {self.task_type}\n\n")
            f.write(md_table)
            f.write("\n\n---\n")
            
        self.logger.info(f"Report appended to {report_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        rep = Reporter(sys.argv[1])
        rep.save_report()
    else:
        print("Usage: python reporter.py <run_directory>")
