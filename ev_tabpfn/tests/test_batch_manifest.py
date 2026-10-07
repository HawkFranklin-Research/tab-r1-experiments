from __future__ import annotations

import json

import pandas as pd

from ev_tabpfn import evaluate_batch


def test_batch_manifest_records_success(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": list(range(40)), "label": [0, 1] * 20}).to_csv(path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "test_batch",
                "output_root": str(tmp_path / "outputs"),
                "models": {
                    "tabpfn": {"enabled": False},
                    "autogluon": {"enabled": False},
                    "catboost": {"enabled": False},
                    "xgboost": {"enabled": False},
                    "lightgbm": {"enabled": False},
                    "random_forest": {"enabled": True},
                    "logistic_regression": {"enabled": False},
                },
                "datasets": [{"name": "tiny", "path": str(path), "target_column": "label"}],
            }
        )
    )
    result = evaluate_batch(config_path)
    assert result.counts["datasets_success"] == 1
    assert (tmp_path / "outputs" / "batch_manifest.json").exists()

