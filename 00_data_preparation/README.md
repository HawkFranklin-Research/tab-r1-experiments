# GPT Multiomics Preprocessing Workspace

This folder contains the preprocessing work for organizing the local raw multiomics cohorts:

```text
/home/prime/Documents/g3/c-5/cptac-5
/home/prime/Documents/g3/c-5/tcga-5
```

The goal is to convert heterogeneous raw CPTAC/LinkedOmics and TCGA/cBioPortal files into machine-learning-ready matrices and graph-ready tensors while preserving provenance, clinical targets, sample identity, missingness, and modality metadata.

The main implementation is:

```text
gpt/preprocess_multiomics.py
```

The generated outputs are under:

```text
gpt/processed/
```

## Current Status

The pipeline has been implemented and partially tested on the local raw data.

Completed successfully:

- Raw inventory was generated for `448` raw files.
- Gene reference was generated from GENCODE v44 plus observed TCGA symbols.
- CPTAC/LinkedOmics and TCGA/cBioPortal omics matrices were harmonized.
- Clinical patient/sample/target tables were parsed.
- Train-ready sparse matrices were generated for all five cancer types and two views.
- Shape and sample/feature index alignment checks passed.

Important caveat:

- An initial validation run found clinical leakage-like fields in the saved train-ready matrices, including fields such as `NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT` and `PERSON_NEOPLASM_CANCER_STATUS`.
- The source code was patched to exclude these fields from future train-ready generation.
- The currently saved `gpt/processed/train_ready/*` matrices were created before that patch and should be regenerated before modeling.
- Graph tensor export was attempted but was too slow in the first implementation at `1500` nodes; the graph export code needs optimization before large graph generation should be treated as production-ready.

## Cohorts

The workspace covers five cancer types:

```text
BRCA
LUAD
LSCC
HNSCC
ESCA
```

TCGA study mapping:

```text
BRCA  -> brca_tcga_pan_can_atlas_2018
LUAD  -> luad_tcga_pan_can_atlas_2018
LSCC  -> lusc_tcga_pan_can_atlas_2018
HNSCC -> hnsc_tcga_pan_can_atlas_2018
ESCA  -> esca_tcga_pan_can_atlas_2018
```

Important ESCA note:

- `cptac-5/downloads/ESCA` is not true CPTAC ESCA.
- The local `cptac-5/README.md` states that it is a LinkedOmics TCGA-ESCA fallback because a CPTAC ESCA page was not available.
- The pipeline keeps it under the source label `CPTAC` because that is the local folder structure, but scientifically this must be interpreted as `LinkedOmics_TCGA_ESCA_fallback`.

## Directory Layout

```text
gpt/
  README.md
  preprocess_multiomics.py
  processed/
    qc/
    reference/
    harmonized/
    clinical/
    train_ready/
    graphs/
```

### `processed/qc/`

Quality-control and audit outputs.

Important files:

```text
raw_inventory.csv
raw_inventory.json
harmonize_failures.csv
train_ready_report.csv
validation_report.csv
```

`raw_inventory.csv` contains file-level provenance:

- path
- relative path
- size
- suffix
- line count
- header column count
- SHA-256 checksum

`train_ready_report.csv` records, per cancer/view/modality:

- cancer type
- view
- cohort
- modality
- number of samples
- number of features

`validation_report.csv` records matrix integrity checks:

- target row counts
- unique target patient counts
- train-ready matrix shapes
- alignment of matrix rows/columns with sample/feature indexes
- possible clinical leakage features

### `processed/reference/`

Gene-reference and annotation cache.

Important files:

```text
gencode.v44.annotation.gtf.gz
gene_reference.parquet
gene_reference.csv
```

The script downloads GENCODE v44 by default and parses gene-level records. It uses this to map stable Ensembl IDs such as:

```text
ENSG00000000003.15 -> ENSG00000000003 -> approved gene symbol when available
```

The reference table also includes observed TCGA `Hugo_Symbol` and `Entrez_Gene_Id` pairs from the local TCGA expression files.

### `processed/harmonized/`

This is the first organized omics layer.

Structure:

```text
processed/harmonized/
  cptac/
    BRCA/
    LUAD/
    LSCC/
    HNSCC/
    ESCA/
  tcga/
    BRCA/
    LUAD/
    LSCC/
    HNSCC/
    ESCA/
```

For each available modality, three files are written:

```text
{modality}.parquet
{modality}.features.parquet
{modality}.samples.parquet
```

Example:

```text
processed/harmonized/tcga/LUAD/rnaseq_gene.parquet
processed/harmonized/tcga/LUAD/rnaseq_gene.features.parquet
processed/harmonized/tcga/LUAD/rnaseq_gene.samples.parquet
```

The `{modality}.parquet` matrix is organized as:

```text
sample_id | feature_1 | feature_2 | ... | feature_n
```

The `{modality}.features.parquet` file records:

- feature ID
- modality
- source path

The `{modality}.samples.parquet` file records:

- canonical sample ID
- raw sample ID
- canonical patient ID
- cohort
- cancer type
- tissue type
- modality

### `processed/clinical/`

Clinical and target data.

Important files:

```text
patient_master.parquet
sample_master.parquet
tcga_clinical_samples.parquet
targets.parquet
```

`patient_master.parquet` contains clinical rows from:

- CPTAC `*_meta.txt`
- LinkedOmics ESCA clinical fallback
- TCGA `data_clinical_patient.txt`

`sample_master.parquet` contains canonical sample metadata from harmonized omics and TCGA clinical sample files.

`targets.parquet` contains target columns such as:

```text
OS_days
OS_event
PFS_days
PFS_event
DSS_days
DSS_event
DFS_days
DFS_event
```

For TCGA, month-based fields are converted to days with:

```text
days = months * 30.4375
```

### `processed/train_ready/`

Machine-learning-ready sparse matrices.

Structure:

```text
processed/train_ready/
  BRCA/
    core/
    proteogenomic/
  LUAD/
    core/
    proteogenomic/
  LSCC/
    core/
    proteogenomic/
  HNSCC/
    core/
    proteogenomic/
  ESCA/
    core/
    proteogenomic/
```

Each view contains:

```text
X.npz
missing_mask.npz
sample_index.csv
feature_index.csv
splits.json
```

`X.npz`:

- Sparse CSR feature matrix.
- Rows are samples.
- Columns are features.
- Continuous values are imputed and z-scored.
- Binary values are imputed with zero and kept on their native scale.

`missing_mask.npz`:

- Sparse CSR missingness mask.
- Same shape as `X.npz`.
- Intended for models that explicitly use missingness.

`sample_index.csv`:

- Row index metadata.
- Includes sample ID, patient ID, cohort, cancer type, tissue, and target columns.

`feature_index.csv`:

- Column index metadata.
- Feature names are prefixed by modality:

```text
rnaseq_gene::GENE:TP53
cnv_log2::GENE:EGFR
cnv_gistic::GENE:MYC
mutation_binary::GENE:KRAS
methylation_gene::GENE:CDKN2A
protein_gene::GENE:AKT1
phosphosite::PHOS:GENE:AKT1:S473:...
clinical::AGE
```

`splits.json`:

- Random train/validation/test split.
- Current default: `70% / 15% / 15%`.
- These are basic splits, not final publication-quality splits.

### `processed/graphs/`

Graph outputs are intended to go here.

Current graph export design:

- One graph per sample.
- Gene nodes.
- Node features from aligned omics values.
- Edge structure from correlation-derived gene-gene edges.
- Output file: `graph_tensors.pt`.

Current status:

- The graph export implementation exists, but the first run was too slow at `1500` nodes.
- It should be optimized before being used at scale.
- Suggested optimization: generate a compact gene-by-modality matrix first, avoid sample-wise sparse random access, and optionally reduce to `500-1000` graph nodes for first benchmarks.

## Canonical IDs

The code standardizes sample and patient IDs.

Canonical patient ID:

```text
{COHORT}:{CANCER}:{RAW_PATIENT_ID}
```

Examples:

```text
CPTAC:LUAD:C3L-00001
TCGA:LUAD:TCGA-05-4244
```

Canonical sample ID:

```text
{COHORT}:{CANCER}:{RAW_SAMPLE_ID}
```

Examples:

```text
CPTAC:LUAD:C3L-00001
TCGA:LUAD:TCGA-05-4244-01
```

For TCGA:

- Patient ID is inferred from the first 12 characters of the barcode.
- Tissue type is inferred from the TCGA sample code.
- Codes like `01` are treated as tumor.
- Codes like `10` or `11` are treated as normal.

## Modalities

The pipeline recognizes and organizes these modality types where available:

```text
rnaseq_gene
rnaseq_isoform
circrna
mirna
methylation_gene
cnv_log2
cnv_gistic
mutation_binary
mutation_site
protein_gene
protein_sepep
phosphosite
rppa_gene
rppa_analyte
tcga_protein_gene
tcga_phosphosite
fusion
phenotype
cna_focal
cna_focal_threshold
```

### Core View

The `core` train-ready view is intended to use broadly comparable molecular layers:

```text
rnaseq_gene
cnv_log2
cnv_gistic
mutation_binary
methylation_gene
clinical features
```

This is the preferred starting point for TCGA-to-CPTAC transfer experiments because these data types exist in both cohorts more consistently than proteomics.

### Proteogenomic View

The `proteogenomic` view adds protein-related layers where available:

```text
protein_gene
protein_sepep
phosphosite
rppa_gene
tcga_protein_gene
tcga_phosphosite
```

Important scientific caution:

- CPTAC mass-spec proteomics and TCGA RPPA are not the same assay.
- They should not be interpreted as directly interchangeable measurements.
- The pipeline keeps them as distinct modality-prefixed features.

## Normalization and Missing Data

At the train-ready stage:

- Features must pass a minimum observed fraction threshold.
- Default threshold: `0.50`.
- All-zero and single-value features are removed.
- Continuous features are median-imputed.
- Continuous features are conservatively centered by cohort when multiple cohorts are present.
- Continuous features are z-scored.
- Binary mutation and GISTIC-like data are imputed with zero.
- Missingness is preserved separately in `missing_mask.npz`.

The current implementation does not perform sophisticated ComBat modeling with biological covariates. It uses conservative cohort-centering for continuous features. For publication-grade modeling, a stricter batch-correction stage should be added and evaluated with before/after PCA plots.

## Clinical Targets

The main target file is:

```text
processed/clinical/targets.parquet
```

Primary recommended endpoint:

```text
PFS/PFI
```

Secondary endpoint:

```text
OS
```

Reason:

- OS can be strongly affected by follow-up duration, treatment, and post-progression care.
- PFS/PFI can be more directly connected to disease biology in many TCGA-style analyses.
- TCGA-CDR recommends standardized use of endpoints such as OS, DSS, DFI, and PFI.

Clinical leakage caution:

- Fields that directly encode tumor recurrence, progression, survival status, or post-treatment outcome must not be used as predictors for survival/progression models.
- The code now excludes leakage-like clinical fields such as:

```text
NEW_TUMOR_EVENT
PERSON_NEOPLASM_CANCER_STATUS
TUMOR_STATUS
PRIMARY_THERAPY_OUTCOME
STATUS
VITAL_STATUS
```

Because earlier matrices were generated before this exclusion patch, rerun `train-ready` before modeling.

## Commands

Run the full pipeline:

```bash
python gpt/preprocess_multiomics.py all \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

Run individual stages:

```bash
python gpt/preprocess_multiomics.py inventory \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

```bash
python gpt/preprocess_multiomics.py reference \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

```bash
python gpt/preprocess_multiomics.py harmonize \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

```bash
python gpt/preprocess_multiomics.py clinical \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

```bash
python gpt/preprocess_multiomics.py train-ready \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

```bash
python gpt/preprocess_multiomics.py validate \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

Graph export:

```bash
python gpt/preprocess_multiomics.py graphs \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed \
  --graph-max-nodes 500
```

Use a smaller `--graph-max-nodes` value first because graph export needs optimization.

## Code Structure

The pipeline is intentionally kept in one script for now:

```text
gpt/preprocess_multiomics.py
```

Major functions:

```text
build_inventory()
build_gene_reference()
harmonize_omics()
parse_clinical()
make_train_ready()
build_graph_tensors()
validate_outputs()
```

Important helper concepts:

- `GeneMapper`: maps Ensembl IDs and symbols to canonical feature IDs.
- `read_feature_matrix()`: reads feature-by-sample matrices and converts them to sample-by-feature matrices.
- `derive_tcga_mutation_binary()`: converts TCGA MAF mutation rows into sample-by-gene binary mutation matrices.
- `clinical_feature_matrix()`: creates clinical predictors while excluding target/leakage fields.
- `normalize_part()`: applies modality-aware imputation and scaling.
- `correlation_edges()`: builds graph edges from gene-gene correlations.

## Suggested Scientific Experiments

### 1. Clinical-Only Baseline

Purpose:

- Establish a minimal baseline using known clinical predictors.

Features:

```text
age
sex
stage
tumor size
smoking
ancestry where available
```

Models:

```text
Cox proportional hazards
elastic-net Cox
random survival forest
gradient boosting survival models
```

Metrics:

```text
C-index
integrated Brier score
time-dependent AUC at 1, 3, and 5 years
```

### 2. Single-Omics Baselines

Purpose:

- Determine whether each modality has predictive value by itself.

Run separate models for:

```text
RNA only
methylation only
CNV only
mutation only
protein only
phosphosite only
clinical only
```

This is necessary before claiming that multiomics integration improves prediction.

### 3. Core Multiomics

Purpose:

- Test broadly available modalities shared between TCGA and CPTAC.

Use:

```text
RNA + mutation + CNV + methylation + clinical
```

This should be the first serious model benchmark.

### 4. Proteogenomic Increment

Purpose:

- Test whether protein/phosphoprotein layers improve over core genomics.

Comparison:

```text
core vs proteogenomic
```

This is the most important CPTAC-specific experiment.

Question:

```text
Does proteomics/phosphoproteomics improve survival or progression prediction beyond RNA/CNV/mutation/methylation?
```

### 5. TCGA-to-CPTAC Transfer

Purpose:

- Measure cross-cohort generalization.

Design:

```text
Train: TCGA
Test: CPTAC
Cancer types: BRCA, LUAD, LSCC/LUSC, HNSCC/HNSC
```

This is scientifically stronger than random mixing because it tests whether the model learned biology or cohort-specific artifacts.

### 6. Cancer Holdout

Purpose:

- Measure whether a pan-cancer model generalizes to unseen cancer types.

Design:

```text
Train on four cancer types.
Hold out one cancer type.
Repeat for each cancer.
```

### 7. Graph Autoencoder / Graph Survival Model

Purpose:

- Use the previous graph-autoencoder direction with richer omics features.

Graph design:

```text
nodes = genes
node features = RNA, methylation, CNV, mutation, protein, phosphosite aggregate
edges = PPI network or data-derived correlations
```

Downstream tasks:

```text
latent embedding survival prediction
patient clustering
Kaplan-Meier/log-rank subgroup testing
attention-based biomarker discovery
```

### 8. Ablation and Robustness

Run:

```text
with vs without batch correction
with vs without clinical features
with vs without missingness mask
different missingness thresholds: 0.50, 0.70, 0.90
TCGA only vs CPTAC only vs combined
```

## Literature Benchmarks

These papers and resources are relevant benchmarks for the scientific design and expected comparisons.

### TCGA Clinical Data Resource

Reference:

- Liu et al., 2018. "An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics."

Why it matters:

- Defines standardized TCGA survival endpoints.
- Supports using OS, DSS, DFI, and PFI.
- PFI is often preferred for pan-cancer progression analyses.

Link:

```text
https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018
```

### cBioPortal Data Formats

Why it matters:

- TCGA files here are cBioPortal PanCancer Atlas style.
- Clinical files use `PATIENT_ID`, `SAMPLE_ID`, `OS_STATUS`, `OS_MONTHS`, `PFS_STATUS`, `PFS_MONTHS`, etc.

Link:

```text
https://docs.cbioportal.org/file-formats/
```

### LinkedOmics

Reference:

- Vasaikar et al., 2018. "LinkedOmics: analyzing multi-omics data within and across 32 cancer types."

Why it matters:

- Explains the LinkedOmics-style multiomics source used for CPTAC-like local downloads.
- Relevant for `.cct`, `.cbt`, `.tsi` style files.

Link:

```text
https://pmc.ncbi.nlm.nih.gov/articles/PMC5753188/
```

### ComBat Batch Correction

Reference:

- Johnson, Li, and Rabinovic, 2007. "Adjusting batch effects in microarray expression data using empirical Bayes methods."

Why it matters:

- Standard reference for empirical-Bayes batch correction.
- Useful benchmark method for continuous molecular matrices.

Link:

```text
https://academic.oup.com/biostatistics/article/8/1/118/252073
```

### MOGONET

Reference:

- Wang et al., 2021. "MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification."

Why it matters:

- Important graph neural network benchmark for multiomics classification.
- Useful comparison for graph-based multiomics integration.

Link:

```text
https://www.nature.com/articles/s41467-021-23774-w
```

### MultiSurv

Reference:

- Vale-Silva and Rohr, 2021. "Long-term cancer survival prediction using multimodal deep learning."

Why it matters:

- Relevant deep-learning benchmark for TCGA survival prediction.
- Integrates multiple modalities for prognosis.

Link:

```text
https://www.nature.com/articles/s41598-021-92799-4
```

### General Benchmarking Position

For a credible scientific paper or thesis-style experiment, the graph model should not be compared only against another neural network. It should be benchmarked against:

```text
clinical-only Cox
elastic-net Cox
random survival forest
single-omics models
early-fusion multiomics models
late-fusion multiomics models
MOGONET-like GCN integration for classification tasks
MultiSurv-like multimodal survival prediction for survival tasks
```

If the graph model does not beat or complement these baselines under cross-cohort validation, it is not yet scientifically convincing.

## Known Limitations

1. The saved train-ready matrices should be regenerated after the clinical leakage patch.
2. Graph export needs performance optimization.
3. Current batch correction is conservative cohort-centering, not full ComBat with biological covariates.
4. CPTAC MS proteomics and TCGA RPPA are intentionally kept separate; they are not assay-equivalent.
5. ESCA under `cptac-5` is a LinkedOmics TCGA fallback, not true CPTAC.
6. The train/validation/test split is a basic random split; final experiments should use cancer/cohort-aware splits.
7. Feature spaces are high-dimensional; downstream models must use regularization, feature selection, dimensionality reduction, or graph aggregation.

## Recommended Next Steps

1. Regenerate train-ready matrices after the leakage patch:

```bash
python gpt/preprocess_multiomics.py train-ready \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

2. Rerun validation:

```bash
python gpt/preprocess_multiomics.py validate \
  --project-root /home/prime/Documents/g3/c-5 \
  --out-dir /home/prime/Documents/g3/c-5/gpt/processed
```

3. Start modeling with clinical-only and single-omics baselines before graph models.

4. Optimize graph export before using `processed/graphs/` for model training.
