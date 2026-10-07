# Reproduction instructions

Use the analysis environment recorded in `environment/requirements.txt`.
Historical training environments may differ; archived configurations and
`environment/observed_local_versions.json` distinguish recorded versions.

```bash
python -m pip install -r environment/requirements.txt
python release_tools/rebuild.py --output-root /tmp/tabr1-rebuild
```

The rebuild copies inputs to a separate output location and executes, in order,
the manuscript asset builder, benchmark/rank statistics, fold-aware pooling
contrast and main figure builder. CPU threads are capped at 12 and address
space at 12 GiB. No learner is fitted. `--smoke` uses 10 bootstrap draws for a
path/execution check only; those intervals are not publication estimates.

| Output | Script | Main inputs |
| --- | --- | --- |
| Tables 1-3 and Extended Data 1-5 | `03_analysis_and_figures/build_manuscript_assets.py` | Local/cloud metrics, historical controls, landscape, folds |
| Figure 2 | `reusability/compute_reusability_statistics.py` | Benchmark summaries |
| Figure 3 | Same statistics script and figure builder | Matched cancer metrics |
| Figure 4 | `reusability/build_reusability_figures.py` | Historical shortcut and permutation summaries |
| Figure 5 and Extended Data 6 | `reusability/fold_aware_contrast.py`, figure builder | Saved patient probabilities and test metadata |
| Figure 1 | Figure builder | Cohort table; explanatory vector schematic |

Paths beginning `reusability/` above are under `03_analysis_and_figures/`.

## Optional refitting

Install `environment/requirements-models.txt` from the repository root. Install
TabFM from its upstream version in THIRD_PARTY.md. Model weights are not bundled;
accept applicable licenses and authenticate through supported upstream tools.
GPU execution is optional for compatible configurations, not universally
required: the recorded foundation-model study used CPUs. TabPFN v2-series
large-data CPU safeguards apply; do not silently change the original protocol.

```bash
python 02_cancer_evaluation/prepare_leakage_safe_folds.py --help
python 02_cancer_evaluation/run_leakage_safe_fold_models.py --help
```

Refitting and cloud deployment have not been executed as part of this release.
The historical benchmark scripts require separately obtained datasets; no
unverified-license benchmark CSVs are redistributed here.
