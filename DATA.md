# Data and redistribution

Original molecular and clinical data originate from TCGA and CPTAC resources.
Consult GDC, PDC and cBioPortal and comply with each source's access conditions;
this repository does not confer rights to controlled-access material.

`data/frozen_test_sets` holds the original 400 patient-grouped evaluation splits,
selected features, clinical test metadata and checksum manifests. Predictions
are stored under `results/cancer/local_models` and `cloud_models/tabr1_results`.
Public redistribution of these patient-level artifacts requires the data owner's
confirmation that all inputs are permitted for redistribution and contain no
identifying or controlled-access information. No figshare DOI is claimed until a
deposit exists. The original Hugging Face endpoint is not assumed accessible.

Raw preprocessing expects a project root with `tcga-5` and `cptac-5` downloads;
its documentation describes the inputs and sparse `train_ready` outputs.
Missingness masks denote missing values, not observed zero values.

Historical absolute paths in archived configurations are provenance records;
live scripts resolve inputs through portable repository paths and environment
variables. Do not edit historical probabilities or labels to make comparisons
appear more favorable.
