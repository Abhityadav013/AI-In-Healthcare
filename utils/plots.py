"""
utils/plots.py
Reusable plotting functions for training history, confusion matrix, ROC curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_dir: Path | None = None,
) -> None:
    """Plot and optionally save training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(train_accs, marker="o", label="Train Accuracy")
    axes[0].plot(val_accs, marker="o", label="Val Accuracy")
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(train_losses, marker="o", label="Train Loss")
    axes[1].plot(val_losses, marker="o", label="Val Loss")
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "training_history.png",
                    dpi=150, bbox_inches="tight")
        print(
            f"[plot] Saved training history → {save_dir / 'training_history.png'}")
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: Path | None = None,
) -> None:
    """Plot a seaborn-styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "confusion_matrix.png",
                    dpi=150, bbox_inches="tight")
        print(
            f"[plot] Saved confusion matrix → {save_dir / 'confusion_matrix.png'}")
    plt.show()
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: list[str],
    save_dir: Path | None = None,
) -> None:
    """Plot per-class ROC curves."""
    n_classes = len(class_names)
    y_true_oh = np.eye(n_classes)[y_true]

    colors = ["blue", "green", "orange", "red", "purple", "brown"]
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
        print(f"[plot] Saved ROC curves → {save_dir / 'roc_curves.png'}")
    plt.show()
    plt.close(fig)


def plot_correct_vs_wrong(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_dir: Path | None = None,
) -> None:
    """Bar chart of correct vs. wrong predictions per class."""
    correct = y_true == y_pred
    n_classes = len(class_names)
    correct_counts = np.bincount(y_true[correct], minlength=n_classes)
    wrong_counts = np.bincount(y_true[~correct], minlength=n_classes)

    x = np.arange(n_classes)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, correct_counts, width,
           label="Correct", color="steelblue")
    ax.bar(x + width / 2, wrong_counts,   width,
           label="Wrong",   color="salmon")

    ax.set_ylabel("Number of Samples")
    ax.set_title("Correct vs Wrong Predictions per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "correct_vs_wrong.png",
                    dpi=150, bbox_inches="tight")
        print(
            f"[plot] Saved correct/wrong chart → {save_dir / 'correct_vs_wrong.png'}")
    plt.show()
    plt.close(fig)
