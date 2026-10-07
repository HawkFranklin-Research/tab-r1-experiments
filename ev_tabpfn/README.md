# ev-tabpfn

`ev-tabpfn` is a comprehensive evaluation pipeline for **TabPFN** and other tabular machine learning baselines. It provides a structured way to run, track, and aggregate machine learning experiments on tabular datasets.

This package was designed to facilitate rigorous comparison between TabPFN and industry-standard models like AutoGluon, CatBoost, XGBoost, and LightGBM.

## Key Features

- **Standardized Evaluation**: Consistent train/test splits and metric reporting across all models.
- **Rich Baselines**: Built-in support for AutoGluon, CatBoost, XGBoost, LightGBM, Random Forest, and Logistic Regression.
- **Batch Orchestration**: Run experiments across dozens of datasets with a single JSON configuration.
- **Automated Reporting**: Generates ROC curves, radar plots, and summary Markdown reports.
- **Artifact Management**: Structured output directory for logs, predictions, metrics, and models.
- **CLI & Python API**: Use it as a command-line tool or integrate it into your Python scripts.

## Installation

```bash
pip install ev-tabpfn
```

### Requirements
- Python 3.10+
- Recommended: A fresh Conda environment (Python 3.11 is preferred for best compatibility with AutoGluon).

## Quick Start

### 1. Set your TabPFN Token
To use the latest TabPFN models, you need a token from [TabPFN](https://www.tabpfn.com/).

```bash
export TABPFN_TOKEN="your_actual_tabpfn_token"
```

### 2. Run a Single Dataset Evaluation
Evaluate a single CSV file. Use `--preset smoke` first if you want the fastest sanity check:

```bash
ev-tabpfn run-single --dataset my_data.csv --target target_column --output ./outputs --preset smoke
```

`--output` is the output folder. The evaluator creates `runs/`, `predictions/`, `metrics/`, `plots/`, `metadata/`, and `logs/` inside it.

### Run TabFM

`ev-tabpfn` can also run Google's TabFM when the `tabfm` package and model backend are installed locally:

```bash
ev-tabpfn run-single \
  --dataset my_data.csv \
  --target target_column \
  --task binary \
  --preset tabfm \
  --output ./outputs_tabfm
```

For batch mode, set `"preset": "tabfm"` in the JSON config.

To run the ensemble variant that uses feature crosses, SVD features, NNLS blending, and calibration:

```bash
ev-tabpfn run-single \
  --dataset my_data.csv \
  --target target_column \
  --task binary \
  --preset tabfm-ensemble \
  --output ./outputs_tabfm_ensemble
```

Notes:
- TabFM model weights use Google's TabFM non-commercial license.
- Classification supports up to 10 classes.
- The default package preset uses the JAX backend, single estimator, and a 1024-row training cap for practical local evaluation.
- If CUDA-enabled `jaxlib` is not installed, TabFM runs on CPU and can be slower.

### 3. Run a Batch Evaluation
Run multiple datasets as defined in a configuration file:

```bash
ev-tabpfn run --config config.json
```

### 4. Use Bundled Sample Datasets
The package includes compact smoke-test samples for binary classification, multiclass classification, and regression.

```bash
ev-tabpfn list-samples
ev-tabpfn copy-samples --output ./ev_tabpfn_samples
```

Create a runnable sample config and execute it:

```bash
ev-tabpfn make-sample-config \
  --samples-dir ./ev_tabpfn_samples \
  --output sample_config.json \
  --preset smoke

ev-tabpfn run --config sample_config.json
```

## Required CSV Formats

The evaluator currently supports **single-target tabular CSVs**.

Rules:
- One row equals one sample.
- One column must be the target.
- If `--target` / `target_column` is omitted, the final CSV column is used as the target.
- Feature columns may be numeric or categorical.
- Missing values are handled by baseline preprocessing where supported.
- Multi-output regression and multilabel classification are not currently supported.

Inspect supported formats from the CLI:

```bash
ev-tabpfn data-formats
ev-tabpfn data-formats --task binary
ev-tabpfn data-formats --task multiclass
ev-tabpfn data-formats --task regression
```

Create CSV templates:

```bash
ev-tabpfn make-template --task binary --output binary_template.csv
ev-tabpfn make-template --task multiclass --output multiclass_template.csv
ev-tabpfn make-template --task regression --output regression_template.csv
```

### Binary Classification CSV

Required shape:

```text
feature_1,feature_2,...,target
value,value,...,class_a
value,value,...,class_b
```

Target requirements:
- exactly two unique classes
- labels may be `0/1`, `1/2`, `yes/no`, `bad/good`, or other string labels

### Multiclass Classification CSV

Required shape:

```text
feature_1,feature_2,...,target
value,value,...,class_a
value,value,...,class_b
value,value,...,class_c
```

Target requirements:
- three or more discrete classes
- labels may be strings or integer-like values

### Regression CSV

Required shape:

```text
feature_1,feature_2,...,target
value,value,...,1.23
value,value,...,4.56
```

Target requirements:
- one numeric continuous target column
- single-output regression only

## Minimal Config Generation

For your own CSV, generate a runnable config instead of writing JSON by hand:

```bash
ev-tabpfn validate --dataset my_data.csv --target label

ev-tabpfn make-config \
  --dataset my_data.csv \
  --target label \
  --task binary \
  --preset smoke \
  --output-root ./outputs \
  --output my_config.json

ev-tabpfn run --config my_config.json
```

For `make-config`, `--output` is the config file path and `--output-root` is the evaluation output folder.

Model presets:

```bash
ev-tabpfn presets
```

- `smoke`: fastest local check, sklearn baselines only
- `standard`: GBM/sklearn baselines, no TabPFN or AutoGluon
- `full`: TabPFN, AutoGluon, GBMs, and sklearn baselines
- `tabpfn-generation`: TabPFN-only preset that defaults to TabPFN v3

## TabPFN Generation Comparison

The package can compare TabPFN generations on the same deterministic splits. By default, generation comparison runs `v3`.

```bash
ev-tabpfn compare-generations \
  --datasets data/australian.csv data/car.csv \
  --target target \
  --versions v3 \
  --output ./outputs_generation_compare
```

Compare modern generations:

```bash
ev-tabpfn compare-generations \
  --datasets data/australian.csv data/car.csv \
  --target target \
  --versions v2 v2_5 v2_6 v3 \
  --train-rows-cap 1024 \
  --output ./outputs_generation_compare
```

Run from JSON:

```bash
ev-tabpfn compare-generations --config examples/generation_comparison.json
```

Generation comparison outputs include:

- `predictions/`: prediction CSVs per TabPFN generation
- `raw/`: raw prediction and probability arrays as `.json` and `.npz`
- `metrics/`: per-dataset metrics
- `plots/`: ROC/confusion/comparison plots where available
- `aggregate/generation_mean_metrics.csv`: mean metrics across datasets
- `generation_summary.json`: batch-level status

Supported generation labels:

```text
v1, v2, v2_5, v2_6, v3
```

`v1` requires a legacy TabPFN v1 checkout:

```bash
ev-tabpfn compare-generations \
  --datasets data/australian.csv \
  --target target \
  --versions v1 \
  --legacy-v1-root /path/to/TabPFN_v1 \
  --output ./outputs_generation_v1
```

For normal `run-single` / `run` evaluations, the generic `tabpfn` model now resolves to TabPFN v3 unless a model config overrides it:

```json
{
  "models": {
    "tabpfn": {"enabled": true, "version": "v2_6"},
    "tabpfn_v3": {"enabled": true}
  }
}
```

## Configuration File Structure

The batch evaluation uses a JSON configuration file. Example:

```json
{
  "run_name": "my_experiment",
  "output_root": "./results",
  "seed": 42,
  "run_reports": true,
  "aggregate_after_run": true,
  "models": {
    "tabpfn": {"enabled": true},
    "autogluon": {"enabled": true, "presets": "medium_quality", "time_limit": 60},
    "catboost": {"enabled": true},
    "xgboost": {"enabled": true},
    "lightgbm": {"enabled": true},
    "random_forest": {"enabled": true},
    "logistic_regression": {"enabled": true}
  },
  "datasets": [
    {
      "name": "dataset1",
      "path": "data/dataset1.csv"
    },
    {
      "name": "dataset2",
      "path": "data/dataset2.csv"
    }
  ]
}
```

## Recreating Research Experiments

To recreate the experiments from the original research (e.g., standard classification datasets), follow these steps:

1. **Prepare your environment**:
   ```bash
   conda create -n ev-tabpfn-test python=3.11 -y
   conda activate ev-tabpfn-test
   pip install ev-tabpfn
   ```

2. **Create a configuration file** (e.g., `recreate_benchmark.json`) and list your dataset paths.

3. **Run the batch evaluation**:
   ```bash
   ev-tabpfn run --config recreate_benchmark.json
   ```

4. **Inspect the results**:
   Aggregated results will be available in the `results/` directory under your `output_root`, including:
   - `aggregate_classification.md`: Comprehensive metric comparison.
   - `benchmark_roc_grid.png`: ROC curves for all datasets.
   - `benchmark_summary.md`: High-level summary of model performance.

## Python API

You can also use `ev-tabpfn` programmatically in your Python scripts:

```python
from ev_tabpfn import (
    aggregate_results,
    create_config_template,
    describe_data_formats,
    evaluate_batch,
    evaluate_dataset,
    list_sample_datasets,
)

# Learn required CSV structures
print(describe_data_formats())

# Evaluate a single dataset
evaluate_dataset(
    dataset_path="data.csv",
    target_column="label",
    task="binary",
    output_root="./outputs",
    model_preset="smoke",
)

# Generate a reusable config
create_config_template(
    output_path="config.json",
    dataset_path="data.csv",
    target_column="label",
    task="binary",
    model_preset="smoke",
)

# Run a batch from a config file
evaluate_batch(config_path="config.json")

# Aggregate results from multiple runs
aggregate_results(output_root="./outputs")

# Inspect bundled samples
list_sample_datasets()
```

## CLI Reference

- `ev-tabpfn run`: Run a batch evaluation from a JSON config.
- `ev-tabpfn run-single`: Run evaluation on a single dataset.
- `ev-tabpfn compare-generations`: Compare TabPFN v1/v2/v2.5/v2.6/v3 on shared classification splits.
- `ev-tabpfn aggregate`: Aggregate existing run results into a summary report.
- `ev-tabpfn validate`: Validate dataset format and compatibility.
- `ev-tabpfn summarize-run`: Print a human-readable summary of a specific dataset run.
- `ev-tabpfn generate-report`: Generate visual plots and reports for a run.
- `ev-tabpfn list-samples`: List bundled smoke-test datasets.
- `ev-tabpfn copy-samples`: Copy bundled sample CSVs into a working folder.
- `ev-tabpfn sample-path`: Print the installed path for one bundled sample.
- `ev-tabpfn data-formats`: Describe required CSV structures.
- `ev-tabpfn make-template`: Create a CSV template for a task.
- `ev-tabpfn make-config`: Create a runnable JSON config for one CSV.
- `ev-tabpfn make-sample-config`: Create a runnable JSON config for bundled samples.
- `ev-tabpfn presets`: List model presets.

## PyPI README

This file is the package long description via `pyproject.toml`:

```toml
readme = "README.md"
```

The next PyPI release page will use this README after rebuilding and uploading the next version.

## Output Directory Structure

Each run produces a structured output:

```text
output_root/
├── batch_config.resolved.json  # The final config used
├── batch_manifest.json         # Index of all runs
├── results/                    # Aggregated plots and tables
├── summary/                    # High-level JSON summaries
├── logs/                       # Batch-level logs
└── runs/                       # Individual dataset results
    └── <dataset_name>/
        └── <run_id>/
            ├── predictions/    # CSV predictions per model
            ├── raw/            # Raw predictions/probabilities as JSON and NPZ
            ├── metrics/        # Performance metrics
            ├── plots/          # ROC and PR curves
            └── logs/           # Detailed execution logs
```

## License

See `LICENSE`. Replace the current local placeholder with the final project license before publishing a production release.
