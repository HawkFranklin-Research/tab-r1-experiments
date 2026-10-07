from __future__ import annotations

import numpy as np
import pandas as pd

from ev_tabpfn.evaluation.labels import ClassificationLabelContract


def test_binary_string_labels_probability_columns() -> None:
    contract = ClassificationLabelContract.from_labels("binary", pd.Series(["bad", "good", "bad"]))
    frame = contract.probability_frame(np.array([[0.7, 0.3], [0.2, 0.8]]), ["bad", "good"])
    assert list(frame.columns) == ["prob_bad", "prob_good"]
    assert contract.positive_label == "good"


def test_multiclass_numeric_labels_probability_columns() -> None:
    contract = ClassificationLabelContract.from_labels("multiclass", pd.Series([1, 2, 3, 1]))
    frame = contract.probability_frame(np.array([[0.2, 0.3, 0.5]]), [1, 2, 3])
    assert list(frame.columns) == ["prob_1", "prob_2", "prob_3"]

