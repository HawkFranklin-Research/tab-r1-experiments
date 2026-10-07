# Third-party software and data

Our evaluator is copied from the source working tree, not reconstructed from a
package release. `export_manifest.json` records original file hashes. Recorded
local versions are in `environment/observed_local_versions.json`; upstream
licenses apply independently of this repository's Apache-2.0 license.

TabPFN local installation: version 8.0.3. TabFM upstream source:
https://github.com/google-research/tabfm, recorded working-tree commit prefix
`5ee6cd7`. Upstream model licenses and authentication requirements apply. No
third-party clones, pretrained weights or papers are included.

Benchmark CSVs are not bundled: the source collection's redistribution licenses
have not been established per dataset. Obtain the original TabPFN benchmark
datasets separately and verify their individual licenses. Metric summaries are
included; C-index and fixed-horizon ROC AUC are different estimands.
