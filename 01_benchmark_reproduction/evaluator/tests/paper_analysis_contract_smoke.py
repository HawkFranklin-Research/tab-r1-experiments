from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "paper" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from analyze_saved_cancer_results import binary_metrics  # noqa: E402
from prepare_leakage_safe_folds import endpoint_labels, variance_rank  # noqa: E402
from run_leakage_safe_fold_models import select_threshold  # noqa: E402


class PaperAnalysisContractTest(unittest.TestCase):
    def test_fixed_window_labels_exclude_early_censoring(self) -> None:
        metadata = pd.DataFrame(
            {
                "OS_days": [100, 1200, 500, 2000],
                "OS_event": [1, 1, 0, 0],
            }
        )
        usable, labels, excluded = endpoint_labels(metadata, "os_3yr")
        self.assertEqual(usable.tolist(), [True, True, False, True])
        self.assertEqual(labels.tolist(), [1, 0, 0])
        self.assertEqual(excluded, 1)

    def test_extreme_labels_keep_only_early_death_and_long_survival(self) -> None:
        metadata = pd.DataFrame(
            {
                "OS_days": [100, 1300, 2000, 2200],
                "OS_event": [1, 1, 0, 1],
            }
        )
        usable, labels, excluded = endpoint_labels(metadata, "extreme_os")
        self.assertEqual(usable.tolist(), [True, False, True, False])
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(excluded, 2)

    def test_variance_selection_uses_supplied_training_matrix(self) -> None:
        matrix = sparse.csr_matrix(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 10.0, 1.0],
                    [0.0, 20.0, 1.0],
                ]
            )
        )
        self.assertEqual(variance_rank(matrix, max_features=1).tolist(), [1])

    def test_threshold_is_selected_from_validation_probabilities(self) -> None:
        y_validation = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold = select_threshold(y_validation, probability)
        self.assertGreaterEqual(threshold, 0.4)
        self.assertLessEqual(threshold, 0.6)

    def test_binary_metrics_include_imbalance_and_calibration_metrics(self) -> None:
        metrics = binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
        for key in ("roc_auc", "pr_auc", "balanced_accuracy", "log_loss", "brier"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
