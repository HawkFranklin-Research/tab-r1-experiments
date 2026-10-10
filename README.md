# Tabular foundation model reusability experiments

See RELEASE_STATUS.md for current artifact availability and verification limits.

The frozen test sets and saved results are included under `data/` and `results/`;
the owner has confirmed that their inputs were public, de-identified downloads.
Archived copy: figshare, doi: 10.6084/m9.figshare.34332477
(reserved; the item is intended for publication when the manuscript is accepted).

Code and derived artifacts for evaluating tabular foundation models on small-data
benchmarks and fixed-horizon cancer survival classification. This repository
contains no manuscript, cover letter, literature collection, model weights or
fitted model binaries. The Apache-2.0 license applies to our code, not to upstream
data, software or pretrained weights.

## Organization

| Directory | Role |
| --- | --- |
| `00_data_preparation` | Raw TCGA/CPTAC preprocessing |
| `ev_tabpfn` | Evaluation package, wrappers and tests |
| `01_benchmark_reproduction` | Historical benchmark and generation evaluators |
| `02_cancer_evaluation` | Frozen-fold preparation, model evaluation and controls |
| `03_analysis_and_figures` | Saved-prediction analysis and figure generation |
| `results` | Predictions, metric summaries, figure source data and reference outputs |
| `data/frozen_test_sets` | Original patient-grouped fold bundles |
| `release_tools` | Curated export and resource-limited verification |

Paths are defined in `tabr1_paths.py`. `TABR1_DATA_ROOT` points to the raw-data
project, `TABR1_TRAIN_READY` can override processed inputs, and
`TABR1_FROZEN_TEST_SETS` can override the frozen-fold location. Models are not
refitted when rebuilding the figures. See REPRODUCE.md and DATA.md.

Figure 2 uses the second benchmark run (`run2_batch_s2`). The earlier
`run1_satya_recreation` is retained as historical evidence, not substituted into
the reported comparison. Shortcut controls are historical split diagnostics,
not fold-matched causal attribution. The patient-averaged pooling estimator is
retained solely for the estimator-sensitivity disclosure; the fold-aware script
produces the current pooling comparison. NA configurations remain NA.
