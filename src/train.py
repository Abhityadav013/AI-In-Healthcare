"""
src/train.py
Training script for the Alzheimer's MRI classification model.

Run:
    python src/train.py

What it does:
  1. Loads config, seeds, selects device (MPS / CUDA / CPU).
  2. Builds DataLoaders.
  3. Builds the EfficientNetB0 model.
  4. Trains with:
       - AdamW optimizer + L2 weight decay (replaces L2 regularizer in Keras)
       - ReduceLROnPlateau scheduler
       - Early stopping (saves the best checkpoint automatically)
       - CSV log of epoch-level metrics
  5. Loads the best checkpoint and runs a quick test-set evaluation.
"""

from __future__ import annotations
from utils.seed import set_seed
from utils.plots import plot_training_history
from utils.device import get_device
from src.model import build_model
from src.data_loader import build_dataloaders
from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATA_DIR,
    EPOCHS,
    ES_MIN_DELTA,
    ES_PATIENCE,
    FINAL_MODEL_PATH,
    L2_REG,
    LEARNING_RATE,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    CLASS_NAMES,
)

import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ─── make sure project root is on the path ────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Args:
        patience:   Epochs to wait without improvement.
        min_delta:  Minimum change in val_loss to count as improvement.
        checkpoint: Path to save the best model weights.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        checkpoint: Path = CHECKPOINT_PATH,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint = checkpoint
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save(model)
        else:
            self.counter += 1
            print(
                f"  [early_stopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.stop = True
                print("  [early_stopping] Stopping training early.")

    def _save(self, model: nn.Module) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint)
        print(f"  [checkpoint] Best model saved → {self.checkpoint}")


# ─── One epoch ────────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float]:
    """
    Run one pass over `loader`.

    If `optimizer` is provided → training mode.
    Otherwise → evaluation mode (no grad).

    Returns:
        (avg_loss, accuracy)  both as Python floats.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    data_dir: Path = DATA_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Full training pipeline.

    Args:
        data_dir:   Root of the dataset (contains class sub-folders).
        epochs:     Maximum number of training epochs.
        batch_size: Mini-batch size.
        lr:         Initial learning rate.
        seed:       Random seed for reproducibility.
    """
    set_seed(seed)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    # num_workers=0 is safest on macOS (avoids multiprocessing fork issues with MPS)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # CrossEntropyLoss = LogSoftmax + NLLLoss (equivalent to sparse_categorical_crossentropy)
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW with weight_decay is the PyTorch equivalent of Adam + L2 regulariser
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=L2_REG)

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=ES_PATIENCE,
        min_delta=ES_MIN_DELTA,
        checkpoint=CHECKPOINT_PATH,
    )

    # ── CSV Logger ────────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUTS_DIR / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "train_acc",
                    "val_loss", "val_acc", "lr"])

    # ── History ───────────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs,   val_accs = [], []

    print(f"\n[train] Starting training for up to {epochs} epochs...")
    print(f"[train] Device: {device}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer)
        val_loss,   val_acc = _run_epoch(
            model, val_loader,   criterion, device)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>3}/{epochs}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.2e}  |  {elapsed:.1f}s"
        )

        # Log to CSV
        writer.writerow([epoch, train_loss, train_acc,
                        val_loss, val_acc, current_lr])
        log_file.flush()

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.stop:
            break

    log_file.close()
    print(f"\n[train] Training complete.  Log saved → {log_path}")

    # ── Save final model ──────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"[train] Final model saved → {FINAL_MODEL_PATH}")

    # ── Plot training history ─────────────────────────────────────────────────
    plot_training_history(
        train_losses, val_losses,
        train_accs,   val_accs,
        save_dir=OUTPUTS_DIR,
    )

    # ── Quick test evaluation ─────────────────────────────────────────────────
    print("\n[train] Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, device)
    print(
        f"[train] Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")
    print("\n✓ Run `python src/evaluate.py` for full evaluation metrics and plots.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
