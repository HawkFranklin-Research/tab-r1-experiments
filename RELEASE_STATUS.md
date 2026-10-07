# Release status

The initial public commit contains code and documentation only. Saved results,
historical patient-level inputs and frozen folds are assembled locally but are
withheld from public upload pending the data owner's redistribution confirmation.
Until those artifacts are uploaded, the saved-prediction rebuild requires the
separate local release bundle and is not a self-contained public reproduction.

Local verification: 49 CSV tables compared with no differences in checked
deterministic columns (absolute numeric tolerance 1e-9). Bootstrap uncertainty
columns and historical path metadata were excluded from this smoke comparison.
Complete figure/statistics smoke rebuild with 10 bootstrap
draws; 4,325 saved model/test-set prediction sets loaded; seven selected evaluator
contract tests passed; Python sources compile. No model was fitted. Publication
bootstrap intervals were not replaced by smoke estimates. The full 2,000-draw
numerical-equivalence audit and a fresh environment/container build remain pending.

Excluded beyond the core working scripts: deployment/VM teardown shell scripts
coupled to the original cloud project; Colab notebook; unverified-license raw
benchmark CSVs; uncapped benchmark outputs not consumed by Figure 2; external
model/source clones. The historical controls' inputs are in the local release
bundle, not in the initial public code commit.

The analysis Dockerfile is a rebuild recipe, not a completed Code Ocean capsule.
No figshare upload or DOI is claimed. Original repository visibility is unchanged.
