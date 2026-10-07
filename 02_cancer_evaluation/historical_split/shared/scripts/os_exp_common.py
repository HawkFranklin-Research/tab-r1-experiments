from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_INPUT_ROOT = Path("/home/prime/Documents/g3/c-5/gpt/processed/train_ready")
DEFAULT_PACKAGE_SRC = Path("/home/prime/Documents/g3/tab-r1/package/src")
WINDOWS_DAYS = {"os_3yr": 3 * 365, "os_5yr": 5 * 365}
EXTREME_EARLY_DEATH_DAYS = 3 * 365
EXTREME_LONG_SURVIVAL_DAYS = 5 * 365


@dataclass
class ExportRecord:
    name: str
    path: str
    task: str
    target: str
    source_cancers: list[str]
    view: str
    endpoint: str
    samples: int
    features: int
    class_counts: dict[str, int]
    label_rule: str
    metadata_path: str | None = None
    horizon_days: int | None = None
    excluded_ambiguous: int | None = None
    cancer_counts: dict[str, int] | None = None
    cancer_label_counts: dict[str, dict[str, int]] | None = None
    include_cancer_feature: bool = False
    note: str = ""


def discover_cancers(input_root: Path, view: str) -> list[str]:
    return sorted(path.name for path in input_root.iterdir() if (path / view).is_dir())


def read_train_ready(input_root: Path, cancer: str, view: str) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    base = input_root / cancer / view
    X = sparse.load_npz(base / "X.npz").tocsr().astype(np.float32)
    sample_index = pd.read_csv(base / "sample_index.csv")
    feature_index = pd.read_csv(base / "feature_index.csv")
    if X.shape[0] != len(sample_index):
        raise ValueError(f"{cancer}/{view}: X rows={X.shape[0]} sample_index rows={len(sample_index)}")
    if X.shape[1] != len(feature_index):
        raise ValueError(f"{cancer}/{view}: X columns={X.shape[1]} feature_index rows={len(feature_index)}")
    return X, sample_index, feature_index


def modality_from_feature_id(feature_id: Any) -> str:
    text = str(feature_id)
    if "::" in text:
        return text.split("::", 1)[0]
    if ":" in text:
        return text.split(":", 1)[0]
    if "__" in text:
        return text.split("__", 1)[0]
    return "unknown"


def nonclinical_feature_positions(feature_index: pd.DataFrame) -> np.ndarray:
    feature_ids = feature_index["feature_id"].astype(str)
    return np.flatnonzero((~feature_ids.str.startswith("clinical::")).to_numpy())


def top_variance_positions(X: sparse.csr_matrix, candidates: np.ndarray, max_features: int) -> np.ndarray:
    if len(candidates) == 0:
        raise ValueError("No candidate features available after filtering clinical features.")
    if len(candidates) <= max_features:
        return candidates
    Xc = X[:, candidates]
    mean = np.asarray(Xc.mean(axis=0)).ravel()
    mean_sq = np.asarray(Xc.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0.0)
    selected_local = np.argsort(variance)[-max_features:][::-1]
    return candidates[selected_local]


def safe_feature_name(name: Any) -> str:
    text = str(name)
    return text.replace("::", "__").replace(":", "_").replace(" ", "_").replace("/", "_")


def dense_frame(X: sparse.csr_matrix, feature_index: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    matrix = X[:, positions].toarray().astype(np.float32, copy=False)
    names = feature_index.iloc[positions]["feature_id"].astype(str).tolist()
    return pd.DataFrame(matrix, columns=[safe_feature_name(name) for name in names])


def fixed_window_label(sample_index: pd.DataFrame, horizon_days: int) -> tuple[pd.Series, pd.Series, int]:
    os_days = pd.to_numeric(sample_index["OS_days"], errors="coerce")
    os_event = pd.to_numeric(sample_index["OS_event"], errors="coerce")
    has_survival = os_days.notna() & os_event.notna()

    event_by_horizon = has_survival & (os_event == 1) & (os_days <= horizon_days)
    known_no_event_by_horizon = has_survival & (
        ((os_event == 0) & (os_days >= horizon_days)) | ((os_event == 1) & (os_days > horizon_days))
    )
    usable = event_by_horizon | known_no_event_by_horizon
    excluded_ambiguous = int((has_survival & ~usable).sum())
    y = pd.Series(np.where(event_by_horizon[usable], "1", "0"), index=sample_index.index[usable])
    return usable, y.reset_index(drop=True), excluded_ambiguous


def extreme_survival_label(sample_index: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    os_days = pd.to_numeric(sample_index["OS_days"], errors="coerce")
    os_event = pd.to_numeric(sample_index["OS_event"], errors="coerce")
    has_survival = os_days.notna() & os_event.notna()

    early_death = has_survival & (os_event == 1) & (os_days < EXTREME_EARLY_DEATH_DAYS)
    long_survival = has_survival & (os_event == 0) & (os_days >= EXTREME_LONG_SURVIVAL_DAYS)
    usable = early_death | long_survival
    excluded = int((has_survival & ~usable).sum())
    y = pd.Series(np.where(early_death[usable], "1", "0"), index=sample_index.index[usable])
    return usable, y.reset_index(drop=True), excluded


def class_counts(values: Any) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(values).value_counts().to_dict().items()}


def cancer_label_counts(cancers: pd.Series, labels: pd.Series) -> dict[str, dict[str, int]]:
    frame = pd.DataFrame({"cancer": cancers.astype(str), "target": labels.astype(str)})
    result: dict[str, dict[str, int]] = {}
    for cancer, group in frame.groupby("cancer"):
        result[str(cancer)] = class_counts(group["target"])
    return result


def metadata_frame(sample_index: pd.DataFrame, cancer: str, target: pd.Series) -> pd.DataFrame:
    columns = [col for col in ["sample_id", "patient_id", "cohort_x", "cancer_x", "OS_days", "OS_event"] if col in sample_index.columns]
    frame = sample_index.loc[target.index if target.index.equals(sample_index.index) else sample_index.index[: len(target)], columns].copy()
    frame = frame.reset_index(drop=True)
    frame["cancer_type"] = cancer
    frame["target"] = target.reset_index(drop=True).astype(str)
    return frame


def write_manifest(output_dir: Path, payload: dict[str, Any], records: list[ExportRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {**payload, "records": [asdict(record) for record in records]}
    (output_dir / "manifest.json").write_text(json.dumps(serializable, indent=2, default=str))
    pd.DataFrame([asdict(record) for record in records]).to_csv(output_dir / "manifest.csv", index=False)
