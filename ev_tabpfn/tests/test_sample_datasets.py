from __future__ import annotations

from pathlib import Path

from ev_tabpfn import copy_sample_datasets, get_sample_dataset_path, list_sample_datasets
from ev_tabpfn.data.loader import DataLoader


def test_sample_manifest_and_paths() -> None:
    samples = list_sample_datasets()
    assert {item["name"] for item in samples} == {
        "australian_sample",
        "car_sample",
        "linear_relation_2d_sample",
    }
    for item in samples:
        path = Path(get_sample_dataset_path(item["name"]))
        assert path.exists()
        assert path.name == item["file"]


def test_copy_sample_datasets(tmp_path) -> None:
    copied = copy_sample_datasets(tmp_path)
    assert set(copied) == {"australian_sample", "car_sample", "linear_relation_2d_sample"}
    for path in copied.values():
        assert Path(path).exists()


def test_sample_datasets_load_with_expected_tasks() -> None:
    loader = DataLoader(seed=42)
    for item in list_sample_datasets():
        dataset = loader.load_local_csv(
            get_sample_dataset_path(item["name"]),
            target_column=item["target_column"],
            task_override=item["task"],
        )
        assert dataset.task_type == item["task"]

