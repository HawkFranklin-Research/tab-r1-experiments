# Saved results and source data

These are archived evaluation outputs, not results from the release smoke tests.
Public redistribution was approved by the owner on 11 October 2026. See
`../data/README.md` for the public TCGA/CPTAC sources and de-identification.

| Directory | Contents and use |
| --- | --- |
| `benchmark/run2_batch_s2` | Classical benchmark summaries used for Figure 2 |
| `benchmark/run1_satya_recreation` | Earlier benchmark run retained as provenance |
| `benchmark/tabpfn_generations` | Capped generation comparison used for Figure 2 |
| `benchmark/tabfm` | TabFM benchmark summary used for Figure 2 |
| `cancer/local_models` | Classical model-fold metrics and per-patient predictions |
| `cancer/cloud_models` | Foundation-model and AutoGluon metrics and predictions |
| `cancer/shortcut_controls` | Historical single-split shortcut diagnostics |
| `cancer/stress_tests` | Historical permutation and cohort-held-out diagnostics |
| `cancer/landscape` | Cohort and feature-space summaries |
| `cancer/provenance` | Input/output provenance manifests |
| `source_data` | Quantitative panel inputs, including the labelled legacy estimator |
| `tables` | Manuscript table CSVs |
| `figures` | Reference figure outputs, not manuscript PDFs |

Historical controls were not evaluated on the frozen repeated folds. The legacy
patient-averaged estimator is retained for the estimator-sensitivity disclosure,
not as the current within-cancer pooling estimate. Unavailable configurations
remain NA; no metrics or predictions have been imputed.

Fitted model binaries and model weights are excluded. For refitting commands see
`../REPRODUCE.md`. The saved-prediction rebuild does not require refitting.
Deployment identifiers, networking and financial fields were omitted from
`cancer/cloud_models/EXECUTION_METADATA.json`; its remaining workload metadata
is an archival record, not an artifact-level completeness assertion. Historical
configurations and manifests preserve the original absolute paths. Live scripts
use `tabr1_paths.py` to resolve repository-local inputs.
