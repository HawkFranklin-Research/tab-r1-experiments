from __future__ import annotations

import json

from ev_tabpfn import (
    create_config_template,
    create_csv_template,
    create_sample_config,
    describe_data_formats,
    format_help_text,
)
from ev_tabpfn.config import BatchEvaluationConfig


def test_describe_data_formats_contains_supported_tasks() -> None:
    specs = describe_data_formats()
    assert set(specs) == {"binary", "multiclass", "regression"}
    assert "target" in specs["binary"]
    assert "single-output" in specs["regression"]["summary"].lower()
    assert "Task: binary" in format_help_text("binary")


def test_create_csv_template(tmp_path) -> None:
    path = tmp_path / "binary_template.csv"
    created = create_csv_template("binary", path)
    assert created == str(path.resolve())
    text = path.read_text()
    assert "label" in text
    assert len(text.splitlines()) >= 2


def test_create_config_template_with_smoke_preset(tmp_path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("x,label\n1,0\n2,1\n")
    config_path = tmp_path / "config.json"
    create_config_template(
        output_path=config_path,
        dataset_path=dataset,
        target_column="label",
        task="binary",
        output_root=tmp_path / "outputs",
        model_preset="smoke",
    )
    payload = json.loads(config_path.read_text())
    config = BatchEvaluationConfig.from_dict(payload)
    assert config.datasets[0].task == "binary"
    assert config.models["random_forest"]["enabled"] is True
    assert config.models["tabpfn"]["enabled"] is False


def test_create_sample_config(tmp_path) -> None:
    config_path = tmp_path / "sample_config.json"
    create_sample_config(output_path=config_path, samples_dir=tmp_path / "samples", model_preset="smoke")
    payload = json.loads(config_path.read_text())
    assert len(payload["datasets"]) == 3
    assert payload["models"]["random_forest"]["enabled"] is True

