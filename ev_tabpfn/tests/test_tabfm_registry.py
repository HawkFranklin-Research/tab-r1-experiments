from ev_tabpfn.models.registry import build_models
from ev_tabpfn.models.presets import get_model_preset


def test_tabfm_preset_enables_only_tabfm_foundation_backend():
    preset = get_model_preset("tabfm")
    assert preset["tabfm"]["enabled"] is True
    assert preset["tabfm"]["backend"] == "jax"
    assert preset["tabfm"]["n_estimators"] == 1
    assert preset["tabfm"]["load_kwargs"]["icl_attention_impl"] == "jax"
    assert preset["tabpfn"]["enabled"] is False


def test_tabfm_ensemble_preset_enables_ensemble_mode():
    preset = get_model_preset("tabfm-ensemble")
    assert preset["tabfm"]["enabled"] is True
    assert preset["tabfm"]["ensemble"] is True
    assert preset["tabfm"]["load_kwargs"]["row_attention_impl"] == "jax"


def test_tabfm_registry_builds_spec_without_loading_weights():
    models = {"tabfm": {"enabled": True, "backend": "jax", "max_train_rows": 16}}
    specs = build_models("binary", y_train=[0, 1, 0, 1], models=models)
    assert [spec.name for spec in specs] == ["tabfm"]
    assert specs[0].family == "tabfm"
