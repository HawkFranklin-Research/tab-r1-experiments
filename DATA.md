# Data and redistribution

The owner confirmed public redistribution on 11 October 2026. TCGA inputs came
from cBioPortal PanCancer Atlas 2018 public studies; CPTAC inputs from LinkedOmics
public downloads; survival outcomes from the public TCGA Pan-Cancer Clinical
Data Resource. No login, data-access agreement, identifying information or
controlled-access files were involved, as confirmed by the owner. Upstream terms
still apply; the code licence does not relicense upstream data.

`data/frozen_test_sets` holds the original 400 patient-grouped evaluation splits,
selected features, clinical test metadata and checksum manifests. Predictions
are stored under `results/cancer/local_models` and `cloud_models/tabr1_results`.
These artifacts are included in the GitHub working copy. See `data/README.md`
for layout, provenance and checksum information. Hugging Face is not a release
destination, and its old dataset is not modified.

Archived copy: figshare, doi: (added after deposit).

Raw preprocessing expects a project root with `tcga-5` and `cptac-5` downloads;
its documentation describes the inputs and sparse `train_ready` outputs.
Missingness masks denote missing values, not observed zero values.

Historical absolute paths in archived configurations are provenance records;
live scripts resolve inputs through portable repository paths and environment
variables. Do not edit historical probabilities or labels to make comparisons
appear more favorable.
