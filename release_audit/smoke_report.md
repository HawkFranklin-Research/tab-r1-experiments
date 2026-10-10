# Release verification: 11 October 2026

This is a smoke check, not a full refit or full-bootstrap equivalence audit. The scientific sources received path/import changes only.

| Step | Status | Seconds |
| --- | --- | --- |
| venv | pass | 1.82 |
| install | pass | 1.82 |
| compile | pass | 0.37 |
| tests | failed (unchanged non-path issue) | 8.88 |
| rebuild | pass | 62.23 |
| compare | pass | 0.72 |
| folds_brca | pass | 6.58 |
| classical | pass | 4.48 |
| autogluon | pass | 8.18 |
| foundation | 4 TabPFN pass; TabFM memory allocation failure | 53.78 |
| stress | pass | 2.12 |
| saved | pass | 2.37 |
| landscape | pass | 4.15 |
| provenance | pass | 0.42 |
| benchmark | failed (unchanged non-path issue) | 84.78 |
| preparation_help | pass | 2.48 |
| preparation_import | pass | 2.58 |
| classical_pooled | pass | 4.02 |
| autogluon_pooled | failed (unchanged non-path issue) | 4.92 |
| foundation_pooled | pass | 54.21 |
| Explicit paper-analysis contract tests | 5 passed | 0.77 |
| Cloud entry point | skipped: unconditional HF download and 400-fold/parquet snapshot contract | not run |
| TabFM on pooled mini fold | skipped following memory failure on BRCA | not run |
| Full preprocessing | not attempted: 27 GB raw inputs outside smoke scope; help/import passed | not run |

Combined pytest discovery: 19 passed, 1 failed. Evaluator smoke files do not all match pytest discovery names; the paper-analysis contract file was run explicitly. No tests loading real TabPFN/TabFM weights were in this unit-test selection (generation tests use fakes).

Rebuild: 49 CSVs have no differences in checked deterministic columns (absolute tolerance 1e-9). Bootstrap uncertainty and historical path metadata were excluded. Smoke uses 10 draws, not the archived 2,000; smoke outputs were never substituted for archived results.

Model checks: both mini folds have 80 training, 20 validation and 30 test rows with both classes present. Five classical models and four TabPFN versions passed on each fold. AutoGluon passed on BRCA, with skipped internal learners, but terminated with SIGABRT on the pooled mini fold. TabFM failed on BRCA under the 12 GB limit, so its pooled check was skipped. All 19 successful model outputs have the expected seven prediction columns and result.json. The benchmark was stopped after non-path CUDA allocation and worker errors; no metrics_summary.csv was produced.

Resource limits: 12 thread environment variables and 12 GiB address-space limits on smoke subprocesses. The benchmark process tree and pooled mini-fold checks also had 12-CPU affinity. Process-based parallel backends in the legacy runner encountered errors; their settings were not changed.

Visual check: reference/rebuilt contact sheets inspected for 12 figure outputs. Layouts and deterministic estimates were consistent. Reduced-bootstrap confidence bars differ as expected; this is not pixel or full-interval equivalence.

Frozen files: before/after 3,202 files (2,801 CSV and 401 JSON), all per-file SHA-256 hashes identical. Manifest SHA-256: `f923b021276dfd8d7e8a55544ef48f2b062d02613a95eab1c60a3ab3349cb9cd`.

Cleanup: all scratch smoke inputs, outputs, environments and generated catboost_info/AutogluonModels directories are deleted after this report is written. Original source workspaces and manuscripts were not modified.

## Failure details (not fixed)

### tests
```text
............F.......                                                     [100%]
=================================== FAILURES ===================================
_____________________ test_single_evaluation_smoke_preset ______________________

tmp_path = PosixPath('$REPO/scratch/smoke/pytest/test_single_evaluation_smoke_p0')

    def test_single_evaluation_smoke_preset(tmp_path) -> None:
        path = tmp_path / "data.csv"
        pd.DataFrame({"x": list(range(40)), "label": [0, 1] * 20}).to_csv(path, index=False)
        result = evaluate_dataset(
            dataset_path=str(path),
            target_column="label",
            output_root=str(tmp_path / "outputs_preset"),
            model_preset="smoke",
        )
>       assert result.status == "success"
E       AssertionError: assert 'failed' == 'success'
E
E         - success
E         + failed

ev_tabpfn/tests/test_single_evaluation.py:44: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-10-11 01:40:35,312 | INFO | Starting single-dataset evaluation
2026-10-11 01:40:35,312 | INFO | Dataset path: $REPO/scratch/smoke/pytest/test_single_evaluation_smoke_p0/data.csv
2026-10-11 01:40:35,317 | ERROR | Dataset evaluation failed.
Traceback (most recent call last):
  File "$REPO/ev_tabpfn/src/ev_tabpfn/evaluation/single.py", line 66, in evaluate_dataset
    for spec in build_models(dataset.task_type, dataset.y_train, run_dir=dirs["base"], models=config.models):
                ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "$REPO/ev_tabpfn/src/ev_tabpfn/models/registry.py", line 191, in build_models
    model=LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto"),
          ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
------------------------------ Captured log call -------------------------------
INFO     ev_tabpfn.single.20261011_014035:single.py:23 Starting single-dataset evaluation
INFO     ev_tabpfn.single.20261011_014035:single.py:24 Dataset path: $REPO/scratch/smoke/pytest/test_single_evaluation_smoke_p0/data.csv
ERROR    ev_tabpfn.single.20261011_014035:single.py:164 Dataset evaluation failed.
Traceback (most recent call last):
  File "$REPO/ev_tabpfn/src/ev_tabpfn/evaluation/single.py", line 66, in evaluate_dataset
    for spec in build_models(dataset.task_type, dataset.y_train, run_dir=dirs["base"], models=config.models):
                ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "$REPO/ev_tabpfn/src/ev_tabpfn/models/registry.py", line 191, in build_models
    model=LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto"),
          ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
=========================== short test summary info ============================
FAILED ev_tabpfn/tests/test_single_evaluation.py::test_single_evaluation_smoke_preset
1 failed, 19 passed in 5.73s

```

### autogluon
```text
	Warning: Exception caused CatBoost to fail during training (ImportError)... Skipping this model.
		`import catboost` failed. A quick tip is to install via `pip install autogluon.tabular[catboost]==1.5.0`.
	Warning: Exception caused NeuralNetFastAI to fail during training (ImportError)... Skipping this model.
		Import fastai failed. A quick tip is to install via `pip install autogluon.tabular[fastai]==1.5.0`.
	Warning: Exception caused XGBoost to fail during training... Skipping this model.

XGBoost Library (libxgboost.so) could not be loaded.
Likely causes:
  * OpenMP runtime is not installed
    - vcomp140.dll or libgomp-1.dll for Windows
    - libomp.dylib for Mac OSX
    - libgomp.so for Linux and other UNIX-like OSes
    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.

  * You are running 32-bit Python on a 64-bit OS

Error message(s): ['${HOME}/miniconda3/lib/python3.13/site-packages/xgboost/lib/libxgboost.so: failed to map segment from shared object']

	Warning: Exception caused NeuralNetTorch to fail during training... Skipping this model.
		module 'numpy' has no attribute 'in1d'
Executing 1 fold(s) across 1 model(s) [1 total runs]
[1/1] RUNNING: per_cancer/os_3yr/BRCA/rep0_fold0 -> autogluon...
[1/1] DONE: per_cancer/os_3yr/BRCA/rep0_fold0 -> autogluon [success, AUC=0.3800]

Summary of executed runs:
     scope endpoint cancer  repeat  fold model_name  status
per_cancer   os_3yr   BRCA       0     0  autogluon success

```

### benchmark
```text
2026-10-11 01:43:06,780 | INFO | Starting Phase 4 batch orchestrator
2026-10-11 01:43:06,781 | INFO | Config path: $REPO/scratch/smoke/bench_config.json
2026-10-11 01:43:06,781 | INFO | Launching Phase 2 for australian_sample
2026-10-11 01:44:30,424 | ERROR | Phase 2 failed for australian_sample with exit code -15
INFO:Aggregator:Collecting results for aggregation...
ERROR:Aggregator:No results found to aggregate.
2026-10-11 01:44:30,437 | INFO | Aggregation completed for output_root=$REPO/scratch/smoke/bench
INFO:phase4:Aggregation completed for output_root=$REPO/scratch/smoke/bench
2026-10-11 01:44:30,438 | INFO | Phase 4 finished with summary: {'datasets_total': 1, 'datasets_success': 0, 'datasets_failed': 1, 'datasets_skipped': 0}
INFO:phase4:Phase 4 finished with summary: {'datasets_total': 1, 'datasets_success': 0, 'datasets_failed': 1, 'datasets_skipped': 0}

```

### TabFM memory allocation traceback
```text
Traceback (most recent call last):
  File "$REPO/02_cancer_evaluation/run_leakage_safe_fold_models.py", line 267, in execute_model
    estimator.fit(X_train, y_train)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "$REPO/ev_tabpfn/src/ev_tabpfn/models/tabfm_backend.py", line 73, in fit
    self.estimator_ = self._build_estimator()
                      ~~~~~~~~~~~~~~~~~~~~~^^
  File "$REPO/ev_tabpfn/src/ev_tabpfn/models/tabfm_backend.py", line 42, in _build_estimator
    model = self._load_model()
  File "$REPO/ev_tabpfn/src/ev_tabpfn/models/tabfm_backend.py", line 34, in _load_model
    return tabfm_v1_0_0.load(model_type=model_type, **kwargs)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "${HOME}/Documents/g3/tab-r1/tabfm/tabfm/src/pytorch/tabfm_v1_0_0.py", line 157, in load
    model = TabFM_HF.from_pretrained(HF_REPO_ID, subfolder=model_type)
  File "${HOME}/miniconda3/lib/python3.13/site-packages/huggingface_hub/utils/_validators.py", line 88, in _inner_fn
    return fn(*args, **kwargs)
  File "${HOME}/miniconda3/lib/python3.13/site-packages/huggingface_hub/hub_mixin.py", line 561, in from_pretrained
    instance = cls._from_pretrained(
        model_id=str(model_id),
    ...<5 lines>...
        **model_kwargs,
    )
  File "${HOME}/Documents/g3/tab-r1/tabfm/tabfm/src/pytorch/tabfm_v1_0_0.py", line 96, in _from_pretrained
    return super()._from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~^
        model_id=local_id,
        ^^^^^^^^^^^^^^^^^^
    ...<7 lines>...
        **model_kwargs,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "${HOME}/miniconda3/lib/python3.13/site-packages/huggingface_hub/hub_mixin.py", line 775, in _from_pretrained
    model = cls(**model_kwargs)
  File "${HOME}/Documents/g3/tab-r1/tabfm/tabfm/src/pytorch/model.py", line 440, in __init__
    self.icl_predictor = ICLearning(icl_dim, icl_num_blocks, icl_nhead, max_classes,
                         ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    icl_dim * ff_factor,
                                    ^^^^^^^^^^^^^^^^^^^^
                                    decoder_hidden or icl_dim * 2, is_classifier)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "${HOME}/Documents/g3/tab-r1/tabfm/tabfm/src/pytorch/model.py", line 397, in __init__
    self.tf_icl = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=None)  # ICL has no RoPE
                  ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "${HOME}/Documents/g3/tab-r1/tabfm/tabfm/src/pytorch/model.py", line 194, in __init__
    MultiheadAttentionBlock(d_model, nhead, dim_ff, activation, rope_base)
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "${HOME}/Documents/g3/tab-r1/tabfm/tabfm/src/pytorch/model.py", line 137, in __init__
    self.linear1_gate = nn.Linear(d_model, dim_ff)
                        ~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "${HOME}/miniconda3/lib/python3.13/site-packages/torch/nn/modules/linear.py", line 109, in __init__
    torch.empty((out_features, in_features), **factory_kwargs)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 67108864 bytes. Error code 12 (Cannot allocate memory)

```

### Benchmark underlying errors
```text
2026-10-11 01:43:08,380 | INFO | Starting Phase 2 single-dataset evaluation
2026-10-11 01:43:08,381 | INFO | Dataset path: $REPO/ev_tabpfn/src/ev_tabpfn/datasets/sample/australian_sample.csv
2026-10-11 01:43:10,636 | INFO | Running model: tabpfn
2026-10-11 01:43:11,472 | ERROR | Model failed: tabpfn | CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 3.68 GiB of which 3.48 GiB is free. Including non-PyTorch memory, this process has 154.00 MiB memory in use. Of the allocated memory 55.37 MiB is allocated by PyTorch, and 16.63 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
2026-10-11 01:43:11,472 | INFO | Running model: random_forest
2026-10-11 01:43:11,520 | ERROR | Model failed: random_forest | 'DummyProcess' object has no attribute 'terminate'
2026-10-11 01:43:11,521 | INFO | Running model: logistic_regression

```
