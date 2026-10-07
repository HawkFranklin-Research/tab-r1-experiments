from __future__ import annotations

import json
import logging
from pathlib import Path


class Reporter:
    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.results_dir = self.run_dir.parents[2] / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_metadata = json.loads((self.run_dir / "metadata" / "dataset_metadata.json").read_text())
        self.metrics_summary = json.loads((self.run_dir / "metrics" / "metrics_summary.json").read_text())
        self.task_type = self.dataset_metadata["task_type"]
        self.dataset_name = self.dataset_metadata["dataset_name"]
        self.logger = logging.getLogger("ev_tabpfn.reporter")

    def _format_value(self, val: float | None, std: float | None = None) -> str:
        if val is None:
            return "N/A"
        if std is not None:
            return f"{val:.4f} +/- {std:.4f}"
        return f"{val:.4f}"

    def generate_markdown_table(self) -> str:
        rows = self.metrics_summary.get("rows", [])
        if self.task_type in {"binary", "multiclass"}:
            headers = ["Model", "ROC AUC", "Accuracy", "F1", "Fit Time (s)"]
            table = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
            for row in rows:
                if row["status"] != "success":
                    table.append(f"| {row['model_name']} | FAILED | {row.get('error_type', 'Error')} | - | - |")
                    continue
                table.append(
                    f"| {row['model_name']} | {self._format_value(row.get('roc_auc'))} | "
                    f"{self._format_value(row.get('accuracy'))} | {self._format_value(row.get('f1'))} | "
                    f"{row['fit_time_s']:.3f}s |"
                )
            return "\n".join(table)

        headers = ["Model", "R2", "RMSE", "MAE", "Fit Time (s)"]
        table = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
        for row in rows:
            if row["status"] != "success":
                table.append(f"| {row['model_name']} | FAILED | {row.get('error_type', 'Error')} | - | - |")
                continue
            table.append(
                f"| {row['model_name']} | {self._format_value(row.get('r2'))} | "
                f"{self._format_value(row.get('rmse'))} | {self._format_value(row.get('mae'))} | "
                f"{row['fit_time_s']:.3f}s |"
            )
        return "\n".join(table)

    def save_report(self) -> Path:
        filename_map = {
            "binary": "binary_classification.md",
            "multiclass": "multiclass_classification.md",
            "regression": "regression_single.md",
        }
        report_path = self.results_dir / filename_map.get(self.task_type, "other_results.md")
        with report_path.open("a") as handle:
            handle.write(f"\n## Dataset: {self.dataset_name}\n")
            handle.write(f"Task Type: {self.task_type}\n\n")
            handle.write(self.generate_markdown_table())
            handle.write("\n\n---\n")
        self.logger.info("Report appended to %s", report_path)
        return report_path

