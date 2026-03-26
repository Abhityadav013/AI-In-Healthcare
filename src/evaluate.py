"""
src/evaluate.py
Full evaluation script — generates all metrics and plots.

Run (after training):
    python src/evaluate.py

Outputs (saved to outputs/):
  - confusion_matrix.png
  - roc_curves.png
  - correct_vs_wrong.png
  - classification_report.csv
  - all scalar metrics printed to console
"""

from __future__ import annotations
from utils.seed import set_seed
from utils.plots import (
    plot_confusion_matrix,
    plot_correct_vs_wrong,
    plot_roc_curves,
)
from utils.metrics import compute_all_metrics, print_metrics
from utils.device import get_device
from src.model import build_model
from src.data_loader import build_dataloaders, get_eval_transforms
from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_SIZE,
    OUTPUTS_DIR,
    RANDOM_SEED,
)

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# ─── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── Collect predictions ──────────────────────────────────────────────────────

def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over all batches in `loader` and collect:
      y_true        — ground-truth integer labels
      y_pred        — predicted integer labels
      y_pred_proba  — predicted class probabilities (softmax)
    """
    model.eval()
    all_true, all_pred, all_proba = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            proba = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_true.extend(labels.numpy())
            all_pred.extend(preds)
            all_proba.extend(proba)

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_proba),
    )


# ─── Single-image prediction (for demo / inference) ───────────────────────────

def predict_single_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
) -> tuple[str, float, np.ndarray]:
    """
    Predict the class of a single MRI image.

    Args:
        model:      Loaded AlzheimerClassifier.
        image_path: Path to the .jpg image file.
        device:     Torch device.

    Returns:
        (class_name, confidence, all_probabilities)
    """
    transform = get_eval_transforms(IMAGE_SIZE)

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # (1, C, H, W)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        proba = F.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(proba))
    class_name = CLASS_NAMES[idx]
    confidence = float(proba[idx])

    return class_name, confidence, proba


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(seed: int = RANDOM_SEED) -> None:
    """
    Load the best checkpoint and run full evaluation.
    """
    set_seed(seed)
    device = get_device()

    # ── Model ─────────────────────────────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}.\n"
            "Train the model first:  python src/train.py"
        )

    model = build_model(pretrained=False)   # weights from checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    print(f"[evaluate] Loaded checkpoint from {CHECKPOINT_PATH}")

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=seed,
    )

    # ── Predictions ───────────────────────────────────────────────────────────
    print("\n[evaluate] Running inference on test set...")
    y_true, y_pred, y_proba = collect_predictions(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_all_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    print_metrics(metrics)

    # ── Save classification report ────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_DIR / "classification_report.csv"
    metrics["classification_report"].to_csv(report_path)
    print(f"[evaluate] Classification report saved → {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, save_dir=OUTPUTS_DIR)
    plot_roc_curves(y_true, y_proba, CLASS_NAMES, save_dir=OUTPUTS_DIR)
    plot_correct_vs_wrong(y_true, y_pred, CLASS_NAMES, save_dir=OUTPUTS_DIR)

    print("\n[evaluate] Done.  All outputs saved to:", OUTPUTS_DIR)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
