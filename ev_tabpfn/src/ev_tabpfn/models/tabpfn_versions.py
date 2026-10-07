from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


CURRENT_TABPFN_VERSIONS = {"v2", "v2_5", "v2_6", "v3"}
ALL_TABPFN_VERSIONS = {"v1", *CURRENT_TABPFN_VERSIONS}
DEFAULT_TABPFN_VERSION = "v3"


def normalize_tabpfn_version(version: str | None) -> str:
    if version is None:
        return DEFAULT_TABPFN_VERSION
    cleaned = version.lower().replace(".", "_").replace("-", "_")
    aliases = {
        "1": "v1",
        "2": "v2",
        "25": "v2_5",
        "2_5": "v2_5",
        "26": "v2_6",
        "2_6": "v2_6",
        "3": "v3",
        "tabpfn": DEFAULT_TABPFN_VERSION,
        "tabpfn_v1": "v1",
        "tabpfn_v2": "v2",
        "tabpfn_v2_5": "v2_5",
        "tabpfn_v2_6": "v2_6",
        "tabpfn_v3": "v3",
    }
    normalized = aliases.get(cleaned, cleaned)
    if normalized not in ALL_TABPFN_VERSIONS:
        available = ", ".join(sorted(ALL_TABPFN_VERSIONS))
        raise ValueError(f"Unsupported TabPFN version: {version}. Available versions: {available}")
    return normalized


def _device(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    if config.get("device"):
        return str(config["device"])
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_version_enum(version: str) -> Any:
    from tabpfn.constants import ModelVersion

    mapping = {
        "v2": ModelVersion.V2,
        "v2_5": ModelVersion.V2_5,
        "v2_6": ModelVersion.V2_6,
        "v3": ModelVersion.V3,
    }
    return mapping[version]


def build_tabpfn_classifier(version: str | None = None, config: dict[str, Any] | None = None) -> Any:
    config = config or {}
    normalized = normalize_tabpfn_version(version or config.get("version"))
    if normalized == "v1":
        return build_legacy_v1_classifier(config)

    from tabpfn import TabPFNClassifier

    device = _device(config)
    model_path = config.get("model_path") or config.get("checkpoint")
    if model_path:
        return TabPFNClassifier(model_path=str(Path(model_path).expanduser()), device=device)
    if hasattr(TabPFNClassifier, "create_default_for_version"):
        return TabPFNClassifier.create_default_for_version(_model_version_enum(normalized), device=device)
    if normalized != DEFAULT_TABPFN_VERSION:
        raise RuntimeError("Installed tabpfn runtime does not expose create_default_for_version.")
    return TabPFNClassifier(device=device)


def build_tabpfn_regressor(version: str | None = None, config: dict[str, Any] | None = None) -> Any:
    config = config or {}
    normalized = normalize_tabpfn_version(version or config.get("version"))
    if normalized == "v1":
        raise ValueError("TabPFN v1 support is classification-only.")

    from tabpfn import TabPFNRegressor

    device = _device(config)
    model_path = config.get("model_path") or config.get("checkpoint")
    if model_path:
        return TabPFNRegressor(model_path=str(Path(model_path).expanduser()), device=device)
    if hasattr(TabPFNRegressor, "create_default_for_version"):
        return TabPFNRegressor.create_default_for_version(_model_version_enum(normalized), device=device)
    if normalized != DEFAULT_TABPFN_VERSION:
        raise RuntimeError("Installed tabpfn runtime does not expose create_default_for_version.")
    return TabPFNRegressor(device=device)


def build_legacy_v1_classifier(config: dict[str, Any] | None = None) -> Any:
    config = config or {}
    legacy_root = config.get("legacy_v1_root")
    if not legacy_root:
        raise ValueError("TabPFN v1 requires legacy_v1_root pointing to a TabPFN v1 checkout.")
    legacy_path = str(Path(legacy_root).expanduser().resolve())

    if config.get("allow_runtime_swap"):
        purge_tabpfn_modules()

    existing = sys.modules.get("tabpfn")
    if existing is not None:
        existing_file = str(getattr(existing, "__file__", ""))
        if not existing_file.startswith(legacy_path):
            raise RuntimeError(
                "TabPFN v1 cannot be imported after a different tabpfn runtime is loaded in this Python process. "
                "Run v1 in a separate process or call it before importing modern TabPFN versions."
            )

    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
    from tabpfn import TabPFNClassifier  # type: ignore

    return TabPFNClassifier(
        device=_device(config),
        N_ensemble_configurations=int(config.get("n_ensemble_configurations", 32)),
    )


def purge_tabpfn_modules() -> None:
    for name in list(sys.modules):
        if name == "tabpfn" or name.startswith("tabpfn."):
            del sys.modules[name]


def clear_accelerator_cache() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
