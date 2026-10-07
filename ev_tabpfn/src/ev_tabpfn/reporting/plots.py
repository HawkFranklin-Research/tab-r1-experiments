from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ev_tabpfn.evaluation.labels import ClassificationLabelContract, flatten_predictions

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
except Exception:  # pragma: no cover
    plt = None
    ConfusionMatrixDisplay = None
    RocCurveDisplay = None


def plotting_available() -> bool:
    return plt is not None


def save_classification_plots(
    *,
    output_dir: Path,
    task_type: str,
    model_name: str,
    y_true: Any,
    y_pred: Any,
    y_prob: np.ndarray | None,
    label_contract: ClassificationLabelContract | None = None,
    y_prob_classes: Any | None = None,
) -> list[str]:
    if not plotting_available():
        return []

    saved: list[str] = []
    y_pred = flatten_predictions(y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"{model_name} confusion matrix")
    fig.tight_layout()
    path = output_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path.name)

    if y_prob is not None and task_type == "binary":
        if label_contract is not None:
            y_true_for_roc = label_contract.encode(y_true)
            y_prob_for_roc = label_contract.align_probabilities(y_prob, y_prob_classes)
            positive_index = label_contract.positive_index
        else:
            y_true_for_roc = y_true
            y_prob_for_roc = y_prob
            positive_index = 1
        if y_prob_for_roc is None or positive_index is None:
            return saved
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(
            y_true_for_roc,
            y_prob_for_roc[:, positive_index],
            pos_label=positive_index,
            ax=ax,
        )
        ax.set_title(f"{model_name} ROC curve")
        fig.tight_layout()
        path = output_dir / f"{model_name}_roc_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path.name)

    return saved


def save_regression_plots(*, output_dir: Path, model_name: str, y_true: Any, y_pred: Any) -> list[str]:
    if not plotting_available():
        return []

    saved: list[str] = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.7)
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} actual vs predicted")
    fig.tight_layout()
    path = output_dir / f"{model_name}_actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path.name)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0.0, linestyle="--", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(f"{model_name} residual plot")
    fig.tight_layout()
    path = output_dir / f"{model_name}_residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path.name)

    return saved

