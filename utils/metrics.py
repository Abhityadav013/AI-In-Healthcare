"""
utils/metrics.py
Reusable metric calculations for multi-class classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true:       Ground-truth integer labels, shape (N,)
        y_pred:       Predicted integer labels,    shape (N,)
        y_pred_proba: Predicted class probabilities, shape (N, C)
        class_names:  List of class name strings

    Returns:
        dict with all metric values + a classification_report DataFrame
    """
    # One-hot encode y_true for metrics that need it
    n_classes = len(class_names)
    y_true_oh = np.eye(n_classes)[y_true]

    metrics: dict = {}

    metrics["accuracy"] = float((y_true == y_pred).mean())
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["precision_macro"] = precision_score(
        y_true, y_pred, average="macro")
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true_oh, y_pred_proba)

    # ROC-AUC (one-vs-rest, macro)
    try:
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true_oh, y_pred_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["roc_auc_macro"] = float("nan")

    # Top-2 accuracy
    top2 = np.argsort(y_pred_proba, axis=1)[:, -2:]
    metrics["top2_accuracy"] = float(
        np.any(top2 == y_true[:, None], axis=1).mean()
    )

    # Detailed per-class report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, digits=4
    )
    metrics["classification_report"] = pd.DataFrame(report_dict).transpose()

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict (excluding the DataFrame)."""
    print("\n" + "=" * 50)
    print("  EVALUATION METRICS")
    print("=" * 50)
    scalar_keys = [k for k in metrics if k != "classification_report"]
    for k in scalar_keys:
        print(f"  {k:<22}: {metrics[k]:.4f}")
    print("\n  Classification Report:")
    print(metrics["classification_report"].to_string())
    print("=" * 50 + "\n")
