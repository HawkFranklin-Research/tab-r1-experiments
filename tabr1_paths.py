"""Portable input locations; no workstation paths or credentials."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("TABR1_RESULTS_ROOT", REPO / "results")).expanduser().resolve()
FROZEN = Path(os.environ.get("TABR1_FROZEN_TEST_SETS", REPO / "data/frozen_test_sets")).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("TABR1_DATA_ROOT", REPO / "data/raw")).expanduser().resolve()
TRAIN_READY = Path(os.environ.get("TABR1_TRAIN_READY", DATA_ROOT / "gpt/processed/train_ready")).expanduser().resolve()
