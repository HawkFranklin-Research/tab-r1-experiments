from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2.artifacts import write_predictions
from scripts.phase2.labels import ClassificationLabelContract
from scripts.phase2.metrics import classification_metrics


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "_tmp_phase2_label_contract"
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_labels = pd.Series([1, 2, 1, 2])
    binary_contract = ClassificationLabelContract.from_labels("binary", binary_labels)
    assert binary_contract.positive_label == 2
    assert binary_contract.positive_index == 1

    binary_pred = np.array([[1], [2], [2], [2]])
    binary_prob = np.array(
        [
            [0.80, 0.20],
            [0.10, 0.90],
            [0.40, 0.60],
            [0.25, 0.75],
        ]
    )
    metrics = classification_metrics(
        "binary",
        binary_labels,
        binary_pred,
        binary_prob,
        label_contract=binary_contract,
        y_prob_classes=[1, 2],
    )
    assert metrics["accuracy"] is not None
    assert metrics["f1"] is not None
    assert metrics["roc_auc"] is not None

    binary_predictions_path = output_dir / "binary_predictions.csv"
    write_predictions(
        binary_predictions_path,
        y_true=binary_labels,
        y_pred=binary_pred,
        y_prob=binary_prob,
        label_contract=binary_contract,
        y_prob_classes=[1, 2],
    )
    binary_df = pd.read_csv(binary_predictions_path)
    assert ["prob_1", "prob_2"] == [col for col in binary_df.columns if col.startswith("prob_")]
    assert "y_true_encoded" in binary_df.columns
    assert "y_pred_encoded" in binary_df.columns

    string_labels = pd.Series(["bad", "good", "bad", "good"])
    string_contract = ClassificationLabelContract.from_labels("binary", string_labels)
    assert string_contract.positive_label == "good"
    string_metrics = classification_metrics(
        "binary",
        string_labels,
        np.array(["bad", "good", "good", "good"]),
        binary_prob,
        label_contract=string_contract,
        y_prob_classes=["bad", "good"],
    )
    assert string_metrics["f1"] is not None
    assert string_metrics["roc_auc"] is not None

    multiclass_labels = pd.Series(["acc", "good", "unacc", "vgood"])
    multiclass_contract = ClassificationLabelContract.from_labels("multiclass", multiclass_labels)
    multiclass_prob = np.eye(4)
    multiclass_df = multiclass_contract.probability_frame(
        multiclass_prob,
        y_prob_classes=multiclass_contract.classes,
    )
    assert list(multiclass_df.columns) == ["prob_acc", "prob_good", "prob_unacc", "prob_vgood"]


if __name__ == "__main__":
    main()
