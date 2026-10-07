from __future__ import annotations

import pandas as pd

from ev_tabpfn.data.loader import DataLoader


def test_data_loader_binary_csv(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"x1": range(20), "x2": range(20, 40), "label": [0, 1] * 10}).to_csv(path, index=False)
    dataset = DataLoader(seed=42).load_local_csv(str(path), target_column="label")
    assert dataset.task_type == "binary"
    assert dataset.target_name == "label"
    assert dataset.metadata["total_samples"] == 20

