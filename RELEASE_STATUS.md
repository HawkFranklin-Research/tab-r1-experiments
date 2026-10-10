# Release status

The owner approved public redistribution on 11 October 2026: TCGA inputs from
cBioPortal PanCancer Atlas, CPTAC from LinkedOmics and outcomes from the public
TCGA clinical resource. The release includes frozen folds, saved results and
historical shortcut-control inputs. See DATA.md for the originating terms.

Local verification: 49 CSV tables compared with no differences in checked
deterministic columns (absolute numeric tolerance 1e-9). Bootstrap uncertainty
columns and historical path metadata were excluded from this smoke comparison.
Complete figure/statistics smoke rebuild with 10 bootstrap
draws; 4,325 saved model/test-set prediction sets loaded; Python sources compile. Publication
bootstrap intervals were not replaced by smoke estimates. The full 2,000-draw
numerical-equivalence audit and a fresh environment/container build remain pending.

New smoke checks: five classical models and four cached TabPFN versions passed
on each of two mini folds (80/20/30 rows). AutoGluon passed on BRCA with skipped
internal learners but aborted on the pooled fold. TabFM failed a memory allocation
under the 12 GB limit. The unit suite had 19 passes and one non-path scikit-learn
compatibility failure; five explicitly selected paper-analysis contract tests passed.
BRCA fold construction, stress, saved-results, landscape, provenance and preprocessing
help/import checks passed. The benchmark was stopped after non-path CUDA/worker
errors. The cloud entry point was skipped because it unconditionally downloads
the old HF dataset; no HF request was made. No non-path failure was fixed.

All 3,202 frozen files remained byte-identical; scratch smoke files were deleted.
Sources and manuscripts were not modified. This is not an all-tests-pass release
or a full refit. Details and runtimes: release_audit/smoke_report.md. Source
checksum/path-only review: release_audit/path_only_audit.md.

Excluded beyond the core working scripts: deployment/VM teardown shell scripts
coupled to the original cloud project; Colab notebook; unverified-license raw
benchmark CSVs; uncapped benchmark outputs not consumed by Figure 2; external
model/source clones. Historical inputs are included in the data release.

The analysis Dockerfile is a rebuild recipe, not a completed Code Ocean capsule.
Original repository visibility is unchanged.
Archived copy: figshare, doi: 10.6084/m9.figshare.34332477
(reserved; the item remains private until the owner authorizes publication).
All three archives and the README were uploaded and their sizes and MD5 checksums
verified. CC BY 4.0 was approved by the owner. The editor-only reviewer link is
held outside the public repository. No publication endpoint was called.
