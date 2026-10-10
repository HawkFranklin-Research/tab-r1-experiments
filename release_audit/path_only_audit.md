# Source checksum and path-only audit

All hunks below were reviewed. No model, metric, seed, feature-selection, statistical or plotting logic changes were found.

| File | Changed lines (+ / -) | Original checksum | Purpose |
| --- | --- | --- | --- |
| `00_data_preparation/preprocess_multiomics.py` | 3 / 2 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/evaluator/scripts/phase1/bulk_validate_real_data.py` | 3 / 2 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/evaluator/scripts/phase1/test_data_loader.py` | 3 / 2 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/evaluator/scripts/phase3/aggregator.py` | 3 / 2 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/evaluator/tests/paper_analysis_contract_smoke.py` | 2 / 2 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/tabpfn_generations/run_pfn3_eval.py` | 5 / 5 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/tabpfn_generations/compare_generations.py` | 5 / 5 | verified | Relocate paths/imports |
| `02_cancer_evaluation/prepare_leakage_safe_folds.py` | 12 / 3 | verified | Relocate paths/imports |
| `02_cancer_evaluation/run_leakage_safe_fold_models.py` | 12 / 3 | verified | Relocate paths/imports |
| `02_cancer_evaluation/run_cloud_evaluation.py` | 11 / 2 | verified | Relocate paths/imports |
| `02_cancer_evaluation/run_cohort_stress_tests.py` | 11 / 2 | verified | Relocate paths/imports |
| `02_cancer_evaluation/analyze_saved_cancer_results.py` | 13 / 4 | verified | Relocate paths/imports |
| `02_cancer_evaluation/analyze_cancer_landscape.py` | 13 / 4 | verified | Relocate paths/imports |
| `02_cancer_evaluation/run_matched_baselines.py` | 13 / 11 | verified | Relocate paths/imports |
| `02_cancer_evaluation/build_provenance_manifest.py` | 10 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp01_per_cancer_fixed_window/scripts/export_per_cancer_fixed_window.py` | 2 / 2 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp01_per_cancer_fixed_window/scripts/make_per_cancer_report.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp01_per_cancer_fixed_window/scripts/run_per_cancer_models.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp02_combined_fixed_window/scripts/export_combined_fixed_window.py` | 2 / 2 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp02_combined_fixed_window/scripts/make_combined_report.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp02_combined_fixed_window/scripts/run_combined_models.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp03_combined_extreme_survival/scripts/export_extreme_survival.py` | 2 / 2 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp03_combined_extreme_survival/scripts/make_extreme_survival_report.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/exp03_combined_extreme_survival/scripts/run_extreme_survival_models.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/shared/scripts/audit_feature_modalities.py` | 1 / 1 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/shared/scripts/os_exp_common.py` | 3 / 2 | verified | Relocate paths/imports |
| `02_cancer_evaluation/historical_split/shared/scripts/run_foundation_models.py` | 1 / 1 | verified | Relocate paths/imports |
| `03_analysis_and_figures/build_manuscript_assets.py` | 20 / 11 | verified | Relocate paths/imports |
| `03_analysis_and_figures/within_cancer_pooling_contrast.py` | 12 / 3 | verified | Relocate paths/imports |
| `03_analysis_and_figures/reusability/compute_reusability_statistics.py` | 15 / 6 | verified | Relocate paths/imports |
| `03_analysis_and_figures/reusability/fold_aware_contrast.py` | 12 / 3 | verified | Relocate paths/imports |
| `03_analysis_and_figures/reusability/build_reusability_figures.py` | 14 / 5 | verified | Relocate paths/imports |
| `01_benchmark_reproduction/evaluator/configs/batch_s2.json` | 8 / 8 | not recorded in export manifest; original compared | Relocate paths/imports |
| `01_benchmark_reproduction/evaluator/configs/satya_recreation.json` | 8 / 8 | not recorded in export manifest; original compared | Relocate paths/imports |
| `02_cancer_evaluation/cloud/build_hf_fold_dataset.py` | 8 / 4 | not recorded in export manifest; original compared | Relocate paths/imports |

No non-path hunks were reverted. New release orchestration tools have no upstream scientific-script counterpart; they are not claimed to be checksum-identical scientific sources.

The export manifest stores original basenames, destination paths and hashes, with no absolute home paths.
