# Alzheimer's MRI Classification

A clean, local PyTorch project for classifying Alzheimer's disease severity from brain MRI images using **EfficientNetB0 transfer learning**.  
Fully compatible with **macOS Apple Silicon (M4 Pro)** via PyTorch MPS acceleration.

## Classes

| Label | Description |
|---|---|
| 0 | Non Demented |
| 1 | Very mild Dementia |
| 2 | Mild Dementia |
| 3 | Moderate Dementia |

---

## Project Structure

```
project/
├── config.py                  # All hyperparameters and paths
├── requirements.txt
├── README.md
│
├── data/
│   └── imagesoasis/
│       └── Data/
│           ├── Non Demented/
│           ├── Very mild Dementia/
│           ├── Mild Dementia/
│           └── Moderate Dementia/
│
├── models/                    # Saved checkpoints (auto-created)
├── outputs/                   # Plots, logs, reports (auto-created)
│
├── src/
│   ├── data_loader.py         # Dataset, transforms, DataLoaders
│   ├── model.py               # EfficientNetB0 classifier
│   ├── train.py               # Training loop
│   └── evaluate.py            # Full evaluation + plots
│
└── utils/
    ├── device.py              # MPS / CUDA / CPU selection
    ├── metrics.py             # All evaluation metrics
    ├── plots.py               # Confusion matrix, ROC, history plots
    └── seed.py                # Reproducibility helpers
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch with MPS support

PyTorch's official macOS wheel includes MPS. **Do this first, before other packages:**

```bash
pip install torch torchvision torchaudio
```

> MPS is included automatically on macOS with Apple Silicon.  
> No extra flags needed — `torch.backends.mps.is_available()` will return `True`.

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your dataset

Your dataset should be placed at:

```
data/imagesoasis/Data/
    Non Demented/
    Very mild Dementia/
    Mild Dementia/
    Moderate Dementia/
```

If your path is different, edit `DATA_DIR` in `config.py`.

---

## Running

### Train

```bash
python src/train.py
```

Training will:
- Automatically use MPS (Apple GPU) if available
- Apply patient-aware train/test splitting (prevents data leakage)
- Balance classes to 7,000 samples each
- Save the best model to `models/best_model.pt`
- Save a training CSV log to `outputs/training_log.csv`
- Plot and save training history to `outputs/training_history.png`

### Evaluate

```bash
python src/evaluate.py
```

Evaluation will:
- Load the best checkpoint
- Print all metrics: accuracy, F1, precision, recall, Cohen's Kappa, MCC, log loss, ROC AUC
- Save: confusion matrix, ROC curves, correct/wrong chart, classification report CSV

---

## Configuration

All settings are in `config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `TRAIN_SAMPLES_PER_CLASS` | 7000 | Balanced samples per class (train) |
| `TEST_SAMPLES_PER_CLASS` | 800 | Balanced samples per class (test) |
| `IMAGE_SIZE` | (128, 128) | Input image resolution |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | Initial learning rate (Adamax equivalent) |
| `ES_PATIENCE` | 5 | Early stopping patience |

---

## Mac-specific Notes

- MPS acceleration works out of the box with this codebase — no code changes needed.
- `num_workers=0` is set in DataLoader to avoid macOS multiprocessing issues with MPS.
- `pin_memory=False` is set because MPS does not support pinned memory.
- If you see `RuntimeError: MPS backend out of memory`, reduce `BATCH_SIZE` in `config.py`.

---

## Key Improvements Over the Original Notebook

| Issue in Notebook | Fix Applied |
|---|---|
| Hardcoded Kaggle paths | `pathlib.Path` + `config.py` |
| TensorFlow / Keras only | PyTorch + `timm` (full MPS support) |
| No patient-aware split → data leakage | Patient ID-based splitting |
| Test set oversampled with `random.choices` | Only undersample test set |
| Same `random.sample` called twice on test (cell 22 & 23) | Fixed — single balanced split |
| No validation split inside training loop | Proper train/val/test 3-way split |
| Weights downloaded from Kaggle manually | `timm` downloads ImageNet weights automatically |
| No model checkpointing | `EarlyStopping` saves best `.pt` checkpoint |
| No reproducibility guarantees | `set_seed()` covers Python, NumPy, PyTorch |
| Broken `preprocess_image` in cell 53 | Fixed in `evaluate.py` |

---

## Dataset

This project uses the [OASIS Alzheimer's Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) from Kaggle.

---

## License

MIT