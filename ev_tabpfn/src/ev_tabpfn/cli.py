from __future__ import annotations

import argparse
import json
from pathlib import Path

from ev_tabpfn.api import (
    aggregate_results,
    compare_tabpfn_generations,
    copy_sample_datasets,
    create_config_template,
    create_csv_template,
    create_sample_config,
    format_help_text,
    evaluate_batch,
    evaluate_dataset,
    generate_report,
    get_sample_dataset_path,
    list_model_presets,
    infer_task,
    list_sample_datasets,
    summarize_run,
    validate_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev-tabpfn", description="Portable TabPFN evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a batch config")
    run.add_argument("--config", required=True, help="Path to JSON config")
    run.add_argument("--force", action="store_true", help="Rerun datasets even if manifest says they succeeded")

    generations = subparsers.add_parser("compare-generations", help="Compare TabPFN generations on shared splits")
    generations.add_argument("--config", default=None, help="Optional JSON config for generation comparison")
    generations.add_argument("--datasets", nargs="*", default=None, help="CSV dataset paths")
    generations.add_argument("--target", default=None, help="Target column applied to all datasets unless config overrides it")
    generations.add_argument("--task", choices=["binary", "multiclass"], default=None, help="Classification task override")
    generations.add_argument("--versions", nargs="+", default=["v3"], help="Versions to run: v1 v2 v2_5 v2_6 v3. Default: v3")
    generations.add_argument("--output", default="outputs_generation_compare", help="Output root for generation comparison artifacts")
    generations.add_argument("--seed", type=int, default=42)
    generations.add_argument("--train-rows-cap", type=int, default=1024, help="Deterministic training-row cap; use 0 to disable")
    generations.add_argument("--legacy-v1-root", default=None, help="Path to a TabPFN v1 checkout when running v1")
    generations.add_argument("--models-config", default=None, help="Optional JSON file with per-version model settings")

    single = subparsers.add_parser("run-single", help="Run one dataset")
    single.add_argument("--dataset", required=True, help="Path to CSV dataset")
    single.add_argument("--target", default=None, help="Target column")
    single.add_argument("--task", choices=["binary", "multiclass", "regression"], default=None, help="Task override")
    single.add_argument(
        "--output",
        default="outputs",
        help="Output folder/root where runs/, metrics, predictions, plots, metadata, and logs are saved",
    )
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--run-reports", action="store_true")
    single.add_argument("--models-config", default=None, help="Optional JSON file containing the models object")
    single.add_argument(
        "--preset",
        default=None,
        choices=["smoke", "standard", "full", "tabfm", "tabfm-ensemble", "foundation", "tabpfn-generation"],
        help="Model preset to use when --models-config is omitted",
    )

    agg = subparsers.add_parser("aggregate", help="Aggregate an output folder or runs/results paths")
    agg.add_argument("--output-root", default=None)
    agg.add_argument("--runs-root", default=None)
    agg.add_argument("--results-dir", default=None)

    validate = subparsers.add_parser("validate", help="Validate dataset loading and task inference")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--target", default=None)
    validate.add_argument("--task", choices=["binary", "multiclass", "regression"], default=None)

    infer = subparsers.add_parser("infer-task", help="Infer target task type")
    infer.add_argument("--dataset", required=True)
    infer.add_argument("--target", default=None)

    summarize = subparsers.add_parser("summarize-run", help="Print run metadata, metrics, and model status")
    summarize.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("generate-report", help="Generate Phase 3-style report and plots for a run")
    report.add_argument("--run-dir", required=True)

    list_samples = subparsers.add_parser("list-samples", help="List bundled sample datasets")
    list_samples.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    copy_samples = subparsers.add_parser("copy-samples", help="Copy bundled sample datasets to a folder")
    copy_samples.add_argument("--output", required=True, help="Destination folder")

    sample_path = subparsers.add_parser("sample-path", help="Print the installed path for one sample dataset")
    sample_path.add_argument("--name", required=True, help="Sample dataset name")

    formats = subparsers.add_parser("data-formats", help="Describe supported CSV data formats")
    formats.add_argument("--task", choices=["binary", "multiclass", "regression"], default=None)
    formats.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    template = subparsers.add_parser("make-template", help="Create a CSV template for one task")
    template.add_argument("--task", required=True, choices=["binary", "multiclass", "regression"])
    template.add_argument("--output", required=True)

    make_config = subparsers.add_parser("make-config", help="Create a runnable JSON config for one CSV")
    make_config.add_argument("--dataset", required=True)
    make_config.add_argument("--target", default=None)
    make_config.add_argument("--task", choices=["binary", "multiclass", "regression"], default=None)
    make_config.add_argument("--output", required=True)
    make_config.add_argument(
        "--output-root",
        default="outputs",
        help="Output folder/root written into the generated config; evaluation artifacts will be saved there",
    )
    make_config.add_argument("--run-name", default="ev_tabpfn_run")
    make_config.add_argument("--preset", default="smoke", choices=["smoke", "standard", "full", "tabfm", "tabfm-ensemble", "foundation", "tabpfn-generation"])
    make_config.add_argument("--no-reports", action="store_true")
    make_config.add_argument("--no-aggregate", action="store_true")

    sample_config = subparsers.add_parser("make-sample-config", help="Create a runnable config for copied bundled samples")
    sample_config.add_argument("--samples-dir", required=True)
    sample_config.add_argument("--output", required=True)
    sample_config.add_argument("--output-root", default="outputs_sample")
    sample_config.add_argument("--preset", default="smoke", choices=["smoke", "standard", "full", "tabfm", "tabfm-ensemble", "foundation", "tabpfn-generation"])

    presets = subparsers.add_parser("presets", help="List model presets")
    presets.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    return parser


def _load_models_config(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    return payload.get("models", payload)


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = evaluate_batch(args.config, force=args.force)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.counts.get("datasets_failed", 0) == 0 else 1

    if args.command == "compare-generations":
        if args.config:
            result = compare_tabpfn_generations(datasets=[], config=_load_json(args.config))
        else:
            if not args.datasets:
                raise ValueError("Provide --datasets or --config for compare-generations.")
            train_rows_cap = None if args.train_rows_cap == 0 else args.train_rows_cap
            result = compare_tabpfn_generations(
                datasets=args.datasets,
                versions=args.versions,
                output_root=args.output,
                target_column=args.target,
                task=args.task,
                seed=args.seed,
                train_rows_cap=train_rows_cap,
                legacy_v1_root=args.legacy_v1_root,
                model_configs=_load_models_config(args.models_config),
            )
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.counts.get("datasets_failed", 0) == 0 else 1

    if args.command == "run-single":
        result = evaluate_dataset(
            dataset_path=args.dataset,
            target_column=args.target,
            task=args.task,
            output_root=args.output,
            seed=args.seed,
            run_reports=args.run_reports,
            models=_load_models_config(args.models_config),
            model_preset=args.preset,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status == "success" else 1

    if args.command == "aggregate":
        outputs = aggregate_results(output_root=args.output_root, runs_root=args.runs_root, results_dir=args.results_dir)
        print(json.dumps(outputs, indent=2))
        return 0

    if args.command == "validate":
        result = validate_dataset(args.dataset, target_column=args.target, task=args.task)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "infer-task":
        print(infer_task(args.dataset, target_column=args.target))
        return 0

    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), indent=2))
        return 0

    if args.command == "generate-report":
        print(json.dumps(generate_report(args.run_dir), indent=2))
        return 0

    if args.command == "list-samples":
        samples = list_sample_datasets()
        if args.json:
            print(json.dumps(samples, indent=2))
        else:
            for item in samples:
                print(f"{item['name']}\t{item['task']}\t{item['file']}\t{item['rows']} rows")
        return 0

    if args.command == "copy-samples":
        print(json.dumps(copy_sample_datasets(args.output), indent=2))
        return 0

    if args.command == "sample-path":
        print(get_sample_dataset_path(args.name))
        return 0

    if args.command == "data-formats":
        if args.json:
            from ev_tabpfn.api import describe_data_formats, get_data_format

            payload = get_data_format(args.task) if args.task else describe_data_formats()
            print(json.dumps(payload, indent=2))
        else:
            print(format_help_text(args.task))
        return 0

    if args.command == "make-template":
        print(create_csv_template(args.task, args.output))
        return 0

    if args.command == "make-config":
        print(
            create_config_template(
                output_path=args.output,
                dataset_path=args.dataset,
                target_column=args.target,
                task=args.task,
                output_root=args.output_root,
                run_name=args.run_name,
                model_preset=args.preset,
                run_reports=not args.no_reports,
                aggregate_after_run=not args.no_aggregate,
            )
        )
        return 0

    if args.command == "make-sample-config":
        print(
            create_sample_config(
                output_path=args.output,
                samples_dir=args.samples_dir,
                output_root=args.output_root,
                model_preset=args.preset,
            )
        )
        return 0

    if args.command == "presets":
        payload = list_model_presets()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for name, spec in payload.items():
                enabled = [model for model, cfg in spec["models"].items() if cfg.get("enabled")]
                print(f"{name}\t{spec['description']}\tenabled={','.join(enabled)}")
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
