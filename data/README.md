# Frozen patient-grouped test sets

The authors approved public redistribution on 11 October 2026. TCGA inputs
were downloaded publicly from cBioPortal PanCancer Atlas 2018 studies
(`brca`, `esca`, `hnsc`, `luad` and `lusc_tcga_pan_can_atlas_2018`), CPTAC inputs
from LinkedOmics public downloads, and survival outcomes from the public
TCGA Pan-Cancer Clinical Data Resource. Public de-identified barcodes are retained;
no identifying or controlled-access inputs were used, as confirmed by the owner.
Upstream data retain their originating terms, not the repository's code licence.

`frozen_test_sets/` contains 400 splits: 16 eligible tasks, five repeats and five
outer folds. There are 13 within-cancer tasks and three pooled tasks. Each directory
contains training, validation and test matrices and aligned patient metadata,
selected features and a fold configuration. ESCA contributes only the eligible
3-year task. Clinical times are days from diagnosis, not calendar dates.

Layout: `scope/endpoint/cancer/repeat_XX_fold_XX/`. The CSV and JSON fold manifests
identify every split, row count and feature-selection path. The CSV manifest has
no checksum column; per-file hashes are recorded in `export_manifest.json` at
the repository root (`sha256_original`). Do not rewrite historical manifest paths.

The frozen tree contains 3,202 files: 2,801 CSV and 401 JSON files.
SHA-256 of `frozen_test_sets/fold_manifest.csv`:
`f923b021276dfd8d7e8a55544ef48f2b062d02613a95eab1c60a3ab3349cb9cd`.

GitHub working copy: https://github.com/HawkFranklin-Research/tab-r1-experiments

Archived copy: figshare, doi: 10.6084/m9.figshare.34332477
(reserved; the item is intended for publication when the manuscript is accepted).
