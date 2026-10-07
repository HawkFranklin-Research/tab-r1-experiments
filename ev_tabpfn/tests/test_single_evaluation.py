from __future__ import annotations

import pandas as pd

from ev_tabpfn import evaluate_dataset


def test_single_evaluation_random_forest_only(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": list(range(40, 80)),
            "label": ["bad", "good"] * 20,
        }
    ).to_csv(path, index=False)
    result = evaluate_dataset(
        dataset_path=str(path),
        target_column="label",
        output_root=str(tmp_path / "outputs"),
        models={
            "tabpfn": {"enabled": False},
            "autogluon": {"enabled": False},
            "catboost": {"enabled": False},
            "xgboost": {"enabled": False},
            "lightgbm": {"enabled": False},
            "random_forest": {"enabled": True},
            "logistic_regression": {"enabled": False},
        },
    )
    assert result.status == "success"
    assert result.run_dir is not None


def test_single_evaluation_smoke_preset(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": list(range(40)), "label": [0, 1] * 20}).to_csv(path, index=False)
    result = evaluate_dataset(
        dataset_path=str(path),
        target_column="label",
        output_root=str(tmp_path / "outputs_preset"),
        model_preset="smoke",
    )
    assert result.status == "success"
