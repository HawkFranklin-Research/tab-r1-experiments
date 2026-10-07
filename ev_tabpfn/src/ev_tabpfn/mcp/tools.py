from __future__ import annotations

from pathlib import Path
from typing import Any

from ev_tabpfn.api import (
    aggregate_results,
    copy_sample_datasets,
    create_config_template,
    create_csv_template,
    create_sample_config,
    describe_data_formats,
    evaluate_batch,
    evaluate_dataset,
    format_help_text,
    generate_report,
    get_data_format,
    get_model_preset,
    get_sample_dataset_path,
    infer_task,
    list_sample_datasets,
    list_model_presets,
    summarize_run,
    validate_dataset,
)


def create_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("ev-tabpfn")

    @server.tool()
    def validate_dataset_tool(dataset_path: str, target_column: str | None = None, task: str | None = None) -> dict[str, Any]:
        return validate_dataset(dataset_path, target_column=target_column, task=task)

    @server.tool()
    def infer_task_tool(dataset_path: str, target_column: str | None = None) -> str:
        return infer_task(dataset_path, target_column=target_column)

    @server.tool()
    def evaluate_dataset_tool(
        dataset_path: str,
        target_column: str | None = None,
        task: str | None = None,
        output_root: str = "outputs",
        seed: int = 42,
        run_reports: bool = False,
    ) -> dict[str, Any]:
        return evaluate_dataset(
            dataset_path=dataset_path,
            target_column=target_column,
            task=task,
            output_root=output_root,
            seed=seed,
            run_reports=run_reports,
        ).to_dict()

    @server.tool()
    def evaluate_batch_tool(config_path: str, force: bool = False) -> dict[str, Any]:
        return evaluate_batch(config_path, force=force).to_dict()

    @server.tool()
    def aggregate_results_tool(output_root: str) -> dict[str, str]:
        return aggregate_results(output_root=output_root)

    @server.tool()
    def summarize_run_tool(run_dir: str) -> dict[str, Any]:
        return summarize_run(run_dir)

    @server.tool()
    def generate_report_tool(run_dir: str) -> dict[str, str]:
        return generate_report(run_dir)

    @server.tool()
    def list_runs(output_root: str) -> list[str]:
        runs_root = Path(output_root) / "runs"
        if not runs_root.exists():
            return []
        return [str(path) for path in runs_root.glob("*/*") if path.is_dir()]

    @server.tool()
    def inspect_manifest(output_root: str) -> dict[str, Any]:
        import json

        path = Path(output_root) / "batch_manifest.json"
        if not path.exists():
            return {"status": "missing", "path": str(path)}
        return json.loads(path.read_text())

    @server.tool()
    def list_sample_datasets_tool() -> list[dict[str, Any]]:
        return list_sample_datasets()

    @server.tool()
    def get_sample_dataset_path_tool(name: str) -> str:
        return get_sample_dataset_path(name)

    @server.tool()
    def copy_sample_datasets_tool(output_dir: str) -> dict[str, str]:
        return copy_sample_datasets(output_dir)

    @server.tool()
    def describe_data_formats_tool() -> dict[str, Any]:
        return describe_data_formats()

    @server.tool()
    def get_data_format_tool(task: str) -> dict[str, Any]:
        return get_data_format(task)

    @server.tool()
    def format_help_text_tool(task: str | None = None) -> str:
        return format_help_text(task)

    @server.tool()
    def create_csv_template_tool(task: str, output_path: str) -> str:
        return create_csv_template(task, output_path)

    @server.tool()
    def create_config_template_tool(
        output_path: str,
        dataset_path: str,
        target_column: str | None = None,
        task: str | None = None,
        output_root: str = "outputs",
        run_name: str = "ev_tabpfn_run",
        model_preset: str = "smoke",
    ) -> str:
        return create_config_template(
            output_path=output_path,
            dataset_path=dataset_path,
            target_column=target_column,
            task=task,
            output_root=output_root,
            run_name=run_name,
            model_preset=model_preset,
        )

    @server.tool()
    def create_sample_config_tool(
        output_path: str,
        samples_dir: str,
        output_root: str = "outputs_sample",
        model_preset: str = "smoke",
    ) -> str:
        return create_sample_config(
            output_path=output_path,
            samples_dir=samples_dir,
            output_root=output_root,
            model_preset=model_preset,
        )

    @server.tool()
    def list_model_presets_tool() -> dict[str, Any]:
        return list_model_presets()

    @server.tool()
    def get_model_preset_tool(name: str) -> dict[str, Any]:
        return get_model_preset(name)

    return server
