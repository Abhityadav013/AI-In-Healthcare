"""
src/data_loader.py
Dataset loading, per-class balancing, train/val/test splitting,
and PyTorch DataLoader creation.

Key design decisions vs. the original notebook:
- Paths handled with pathlib — no hardcoded strings.
- Patient-aware splitting: images from the same patient stay in the same split,
  preventing data leakage between train and test sets.
- Balancing is done with reproducible random.choices / random.sample using the
  configured seed.
- Returns standard PyTorch DataLoaders (works with MPS, CUDA, CPU).
- Augmentation is applied only to the training set.
"""

from __future__ import annotations
from config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_SAMPLES_PER_CLASS,
    TEST_SPLIT,
    TRAIN_SAMPLES_PER_CLASS,
    VAL_SPLIT,
)

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Filename parsing ─────────────────────────────────────────────────────────

_FILENAME_PATTERN = re.compile(r"OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+)\.jpg")


def _parse_filename(filename: str) -> tuple[int | None, int | None, int | None, int | None]:
    """
    Extract (patient_id, mr_id, scan_id, layer_id) from an OASIS filename.
    Returns (None, None, None, None) if the filename doesn't match the pattern.
    """
    match = _FILENAME_PATTERN.match(filename)
    if match:
        # type: ignore[return-value]
        return tuple(int(g) for g in match.groups())
    return None, None, None, None


# ─── Path collection ──────────────────────────────────────────────────────────

def collect_paths(data_dir: Path) -> dict[str, list[Path]]:
    """
    Walk the data directory and collect image paths per class.

    Expected layout:
        data_dir/
            Non Demented/
            Very mild Dementia/
            Mild Dementia/
            Moderate Dementia/

    Returns:
        { class_name: [Path, ...], ... }
    """
    paths: dict[str, list[Path]] = {}
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}\n"
                f"Make sure your data is placed at: {data_dir}"
            )
        image_paths = sorted(class_dir.glob("*.jpg"))
        paths[class_name] = image_paths
        print(f"  {class_name:<22}: {len(image_paths):>6} images")
    return paths


# ─── Patient-aware splitting ──────────────────────────────────────────────────

def _split_by_patient(
    image_paths: list[Path],
    test_size: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """
    Split image paths into train/test ensuring no patient appears in both splits.

    Why patient-aware?
    ------------------
    The same patient has multiple MRI slices.  If we split naively by image,
    slices from the same patient can end up in both train and test, making the
    model look better than it is (data leakage).  We split by *patient ID*
    first, then collect all slices for each patient.
    """
    # Group paths by patient_id
    patient_to_paths: dict[int, list[Path]] = {}
    ungrouped: list[Path] = []

    for p in image_paths:
        patient_id, *_ = _parse_filename(p.name)
        if patient_id is None:
            ungrouped.append(p)
            continue
        patient_to_paths.setdefault(patient_id, []).append(p)

    patient_ids = list(patient_to_paths.keys())

    if len(patient_ids) < 2:
        # Fallback: too few patients identified — split by image
        return train_test_split(image_paths, test_size=test_size, random_state=seed)

    train_ids, test_ids = train_test_split(
        patient_ids, test_size=test_size, random_state=seed
    )

    train_paths = [p for pid in train_ids for p in patient_to_paths[pid]]
    test_paths = [p for pid in test_ids for p in patient_to_paths[pid]]

    # Put ungrouped images in train (safe default)
    train_paths.extend(ungrouped)

    return train_paths, test_paths


# ─── Balancing ────────────────────────────────────────────────────────────────

def _balance_paths(
    paths: list[Path],
    target: int,
    seed: int,
) -> list[Path]:
    """
    Under- or over-sample a list of paths to exactly `target` items.

    - Oversample (target > len): random.choices (with replacement)
    - Undersample (target < len): random.sample (without replacement)
    """
    rng = random.Random(seed)
    if len(paths) >= target:
        return rng.sample(paths, target)
    else:
        return rng.choices(paths, k=target)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class AlzheimerDataset(Dataset):
    """
    PyTorch Dataset for the OASIS Alzheimer's MRI image dataset.

    Each item: (image_tensor, label_int)
    Image tensor shape: (3, H, W) — float32, normalized to [0, 1] by default.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        assert len(image_paths) == len(
            labels), "Paths and labels must have the same length."
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open as RGB — some grayscale scans in this dataset need the conversion
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms(image_size: tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """
    Training transforms with light augmentation suitable for MRI scans.

    NOTE: MRI images are medical — aggressive augmentation (colour jitter, heavy
    flips, random erasing) can distort clinically relevant features.  We keep
    augmentations conservative.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),                          # → [0, 1], (C, H, W)
        transforms.Normalize(                           # ImageNet mean/std (good for EfficientNet)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size: tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic transforms for validation and test sets — no augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─── Main entry point ─────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Full pipeline: scan → split → balance → Dataset → DataLoader.

    Args:
        data_dir:    Root directory containing the class sub-folders.
        batch_size:  Mini-batch size for all loaders.
        num_workers: Parallel workers for DataLoader.
                     On macOS / MPS, set to 0 to avoid forking issues.
        seed:        Random seed for reproducibility.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print("\n[data_loader] Scanning dataset...")
    all_paths = collect_paths(data_dir)

    train_image_paths: list[Path] = []
    train_labels:      list[int] = []
    test_image_paths:  list[Path] = []
    test_labels:       list[int] = []

    print("\n[data_loader] Splitting by patient (train / test)...")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        paths = all_paths[class_name]

        # Patient-aware train/test split
        tr_paths, te_paths = _split_by_patient(
            paths, test_size=TEST_SPLIT, seed=seed)

        # Balance train split
        tr_paths = _balance_paths(tr_paths, TRAIN_SAMPLES_PER_CLASS, seed=seed)

        # Balance test split (undersample only — don't oversample the test set)
        te_target = min(TEST_SAMPLES_PER_CLASS, len(te_paths))
        te_paths = _balance_paths(te_paths, te_target, seed=seed)

        train_image_paths.extend(tr_paths)
        train_labels.extend([class_idx] * len(tr_paths))
        test_image_paths.extend(te_paths)
        test_labels.extend([class_idx] * len(te_paths))

        print(
            f"  {class_name:<22}: train={len(tr_paths)}, test={len(te_paths)}"
        )

    # Train / val split (stratified by label so class ratio is preserved)
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_image_paths,
        train_labels,
        test_size=VAL_SPLIT,
        stratify=train_labels,
        random_state=seed,
    )

    print(f"\n[data_loader] Final split sizes:")
    print(
        f"  train={len(tr_paths)}, val={len(val_paths)}, test={len(test_image_paths)}")

    # Build Datasets
    train_ds = AlzheimerDataset(
        tr_paths,             tr_labels,   get_train_transforms())
    val_ds = AlzheimerDataset(
        val_paths,            val_labels,  get_eval_transforms())
    test_ds = AlzheimerDataset(
        test_image_paths,     test_labels, get_eval_transforms())

    # Build DataLoaders
    # pin_memory speeds up host→GPU transfer but is not supported for MPS;
    # we disable it unconditionally here for compatibility.
    common = dict(batch_size=batch_size,
                  num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader = DataLoader(val_ds,   shuffle=False, **common)
    test_loader = DataLoader(test_ds,  shuffle=False, **common)

    return train_loader, val_loader, test_loader
