"""
src/data_loader.py
Dataset loading, patient-aware train/val/test splitting,
training-only balancing, and PyTorch DataLoader creation.

Key design decisions:
- All splits are patient-disjoint.
- Train/val/test are created before any balancing.
- Only the training split is balanced.
- Validation and test transforms are deterministic.
- Split diagnostics assert that there is no leakage by patient, scan, or path.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SAMPLES_PER_CLASS,
    VAL_SPLIT,
)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Filename parsing ─────────────────────────────────────────────────────────

_FILENAME_PATTERN = re.compile(r"OAS1_(\d+)_MR(\d+)_mpr-(\d+)_(\d+)\.jpg")


def _parse_filename(filename: str) -> tuple[int | None, int | None, int | None, int | None]:
    """
    Extract (patient_id, mr_id, scan_id, layer_id) from an OASIS filename.
    Returns (None, None, None, None) if the filename doesn't match the pattern.
    """
    match = _FILENAME_PATTERN.fullmatch(filename)
    if match:
        return tuple(int(g) for g in match.groups())
    return None, None, None, None


def _require_parsed_filename(path: Path) -> tuple[int, int, int, int]:
    patient_id, mr_id, scan_id, layer_id = _parse_filename(path.name)
    if patient_id is None:
        raise ValueError(
            f"Unable to extract patient ID from filename: {path.name}\n"
            "Expected pattern: OAS1_<patient>_MR<mr>_mpr-<scan>_<layer>.jpg"
        )
    return patient_id, mr_id, scan_id, layer_id


def _patient_id(path: Path) -> int:
    return _require_parsed_filename(path)[0]


def _scan_key(path: Path) -> tuple[int, int, int]:
    patient_id, mr_id, scan_id, _ = _require_parsed_filename(path)
    return patient_id, mr_id, scan_id


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

def _group_paths_by_patient(image_paths: list[Path]) -> dict[int, list[Path]]:
    """Group image paths by patient ID. Raises on unparseable filenames."""
    patient_to_paths: dict[int, list[Path]] = {}
    for path in image_paths:
        patient_to_paths.setdefault(_patient_id(path), []).append(path)

    for grouped_paths in patient_to_paths.values():
        grouped_paths.sort()
    return patient_to_paths


def _split_sequence(
    items: list[int],
    holdout_fraction: float,
    seed: int,
    ensure_non_empty_holdout: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Deterministically split a sequence into (keep, holdout).

    If the class is very small, the holdout can be empty instead of falling
    back to image-level splitting, which would leak patients across splits.
    """
    items = list(items)
    if not items:
        return [], []
    if len(items) == 1:
        return items, []

    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    holdout_count = int(round(len(shuffled) * holdout_fraction))
    if ensure_non_empty_holdout and holdout_count == 0:
        holdout_count = 1
    holdout_count = max(0, min(len(shuffled) - 1, holdout_count))

    holdout_items = shuffled[:holdout_count]
    keep_items = shuffled[holdout_count:]
    return keep_items, holdout_items


def _flatten_grouped_paths(
    patient_to_paths: dict[int, list[Path]],
    patient_ids: list[int],
) -> list[Path]:
    return [path for patient_id in patient_ids for path in patient_to_paths[patient_id]]


# ─── Balancing ────────────────────────────────────────────────────────────────

def _balance_paths(
    paths: list[Path],
    target: int,
    seed: int,
) -> list[Path]:
    """
    Under- or over-sample a list of training paths to exactly `target` items.

    - Oversample (target > len): random.choices (with replacement)
    - Undersample (target < len): random.sample (without replacement)
    """
    if target <= 0:
        return []
    if not paths:
        raise ValueError("Cannot balance an empty training split.")

    rng = random.Random(seed)
    if len(paths) >= target:
        return rng.sample(paths, target)
    return rng.choices(paths, k=target)


# ─── Split diagnostics ────────────────────────────────────────────────────────

def _class_counts(labels: list[int]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for label in labels:
        counts[CLASS_NAMES[label]] += 1
    return counts


def _print_class_counts(name: str, labels: list[int]) -> None:
    counts = _class_counts(labels)
    print(f"[data_loader] {name} per-class counts:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name:<22}: {counts[class_name]:>6}")


def _assert_no_overlap(
    name_a: str,
    paths_a: list[Path],
    name_b: str,
    paths_b: list[Path],
) -> None:
    path_overlap = set(paths_a) & set(paths_b)
    patient_overlap = {_patient_id(path) for path in paths_a} & {
        _patient_id(path) for path in paths_b}
    scan_overlap = {_scan_key(path) for path in paths_a} & {
        _scan_key(path) for path in paths_b}

    print(
        f"[data_loader] Overlap {name_a}/{name_b}: "
        f"patients={len(patient_overlap)}, scans={len(scan_overlap)}, paths={len(path_overlap)}"
    )

    assert not patient_overlap, (
        f"Patient leakage detected between {name_a} and {name_b}: "
        f"{sorted(patient_overlap)[:10]}"
    )
    assert not scan_overlap, (
        f"Scan leakage detected between {name_a} and {name_b}: "
        f"{sorted(scan_overlap)[:10]}"
    )
    assert not path_overlap, (
        f"File-path leakage detected between {name_a} and {name_b}: "
        f"{[str(path) for path in sorted(path_overlap)[:10]]}"
    )


def _print_split_debug(
    train_paths: list[Path],
    train_labels: list[int],
    val_paths: list[Path],
    val_labels: list[int],
    test_paths: list[Path],
    test_labels: list[int],
) -> None:
    print("\n[data_loader] Split diagnostics:")
    print(
        f"  train unique patients: {len({_patient_id(path) for path in train_paths})}")
    print(
        f"  val   unique patients: {len({_patient_id(path) for path in val_paths})}")
    print(
        f"  test  unique patients: {len({_patient_id(path) for path in test_paths})}")

    _assert_no_overlap("train", train_paths, "val", val_paths)
    _assert_no_overlap("train", train_paths, "test", test_paths)
    _assert_no_overlap("val", val_paths, "test", test_paths)

    _print_class_counts("train", train_labels)
    _print_class_counts("val", val_labels)
    _print_class_counts("test", test_labels)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class AlzheimerDataset(Dataset):
    """
    PyTorch Dataset for the OASIS Alzheimer's MRI image dataset.

    Each item: (image_tensor, label_int)
    Image tensor shape: (3, H, W).
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

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms(image_size: tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """Training transforms with light augmentation suitable for MRI scans."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
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
    Full pipeline: scan → patient split → training-only balance → Dataset → DataLoader.
    """
    print("\n[data_loader] Scanning dataset...")
    all_paths = collect_paths(data_dir)

    train_image_paths: list[Path] = []
    train_labels: list[int] = []
    val_image_paths: list[Path] = []
    val_labels: list[int] = []
    test_image_paths: list[Path] = []
    test_labels: list[int] = []

    print("\n[data_loader] Splitting by patient (train / val / test)...")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        paths = all_paths[class_name]
        patient_to_paths = _group_paths_by_patient(paths)
        patient_ids = sorted(patient_to_paths)

        train_val_ids, test_ids = _split_sequence(
            patient_ids,
            holdout_fraction=TEST_SPLIT,
            seed=seed + class_idx * 1000,
            ensure_non_empty_holdout=True,
        )
        train_ids, val_ids = _split_sequence(
            train_val_ids,
            holdout_fraction=VAL_SPLIT,
            seed=seed + class_idx * 1000 + 1,
            ensure_non_empty_holdout=False,
        )

        raw_train_paths = _flatten_grouped_paths(patient_to_paths, train_ids)
        class_val_paths = _flatten_grouped_paths(patient_to_paths, val_ids)
        class_test_paths = _flatten_grouped_paths(patient_to_paths, test_ids)

        balanced_train_paths = _balance_paths(
            raw_train_paths,
            TRAIN_SAMPLES_PER_CLASS,
            seed=seed + class_idx * 1000 + 2,
        )

        if not class_val_paths:
            print(
                f"  [warning] {class_name}: validation split has 0 patients. "
                "This class is too small for a leak-free 3-way patient split."
            )

        train_image_paths.extend(balanced_train_paths)
        train_labels.extend([class_idx] * len(balanced_train_paths))
        val_image_paths.extend(class_val_paths)
        val_labels.extend([class_idx] * len(class_val_paths))
        test_image_paths.extend(class_test_paths)
        test_labels.extend([class_idx] * len(class_test_paths))

        print(
            f"  {class_name:<22}: "
            f"patients train/val/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)} | "
            f"images train(raw->balanced)/val/test="
            f"{len(raw_train_paths)}->{len(balanced_train_paths)}/{len(class_val_paths)}/{len(class_test_paths)}"
        )

    print("\n[data_loader] Final split sizes:")
    print(
        f"  train={len(train_image_paths)}, "
        f"val={len(val_image_paths)}, "
        f"test={len(test_image_paths)}"
    )

    _print_split_debug(
        train_image_paths,
        train_labels,
        val_image_paths,
        val_labels,
        test_image_paths,
        test_labels,
    )

    train_ds = AlzheimerDataset(
        train_image_paths, train_labels, get_train_transforms())
    val_ds = AlzheimerDataset(
        val_image_paths, val_labels, get_eval_transforms())
    test_ds = AlzheimerDataset(
        test_image_paths, test_labels, get_eval_transforms())

    common = dict(batch_size=batch_size,
                  num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader
