from __future__ import annotations

import numpy as np
import pandas as pd

from ev_tabpfn.evaluation.generation_preprocessing import TabPFNGenerationPreprocessor, cap_training_rows
from ev_tabpfn.evaluation.generations import compare_tabpfn_generations
from ev_tabpfn.models import registry
from ev_tabpfn.models.tabpfn_versions import normalize_tabpfn_version


class FakeClassifier:
    def fit(self, X, y):
        self.classes_ = sorted(pd.Series(y).unique().tolist())
        return self

    def predict(self, X):
        return np.asarray([self.classes_[idx % len(self.classes_)] for idx in range(len(X))])

    def predict_proba(self, X):
        values = np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))
        return values


def test_normalize_tabpfn_version_aliases() -> None:
    assert normalize_tabpfn_version(None) == "v3"
    assert normalize_tabpfn_version("tabpfn_v2.5") == "v2_5"
    assert normalize_tabpfn_version("3") == "v3"


def test_generation_preprocessor_and_train_cap() -> None:
    X = pd.DataFrame({"num": list(range(20)), "cat": ["a", "b"] * 10})
    y = pd.Series([0, 1] * 10)
    capped = cap_training_rows(X, y, task_type="binary", seed=42, max_rows=8)
    transformed = TabPFNGenerationPreprocessor().fit(capped.X_train).transform(capped.X_train)
    assert capped.rows_original == 20
    assert capped.rows_used == 8
    assert transformed.shape == (8, 2)


def test_registry_exposes_versioned_tabpfn(monkeypatch) -> None:
    monkeypatch.setattr(registry, "build_tabpfn_classifier", lambda version=None, config=None: FakeClassifier())
    specs = registry.build_models(
        "binary",
        y_train=pd.Series([0, 1, 0, 1]),
        models={
            "tabpfn": {"enabled": False},
            "tabpfn_v3": {"enabled": True},
            "random_forest": {"enabled": False},
            "logistic_regression": {"enabled": False},
            "catboost": {"enabled": False},
            "xgboost": {"enabled": False},
            "lightgbm": {"enabled": False},
            "autogluon": {"enabled": False},
        },
    )
    assert [spec.name for spec in specs] == ["tabpfn_v3"]


def test_compare_generations_writes_raw_outputs(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "x": list(range(40)),
            "category": ["a", "b"] * 20,
            "target": ["no", "yes"] * 20,
        }
    ).to_csv(data_path, index=False)

    import ev_tabpfn.evaluation.generations as generations

    monkeypatch.setattr(generations, "build_tabpfn_classifier", lambda version=None, config=None: FakeClassifier())
    result = compare_tabpfn_generations(
        datasets=[str(data_path)],
        versions=["v3"],
        target_column="target",
        output_root=tmp_path / "outputs",
        train_rows_cap=12,
    )

    assert result.counts["datasets_success"] == 1
    assert (tmp_path / "outputs" / "aggregate" / "generation_mean_metrics.csv").exists()
    assert list((tmp_path / "outputs").glob("runs/data/*/raw/tabpfn_v3/raw_predictions.npz"))

