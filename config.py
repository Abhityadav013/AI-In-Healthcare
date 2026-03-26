"""
config.py
Central configuration for the Alzheimer's MRI Classification project.
Edit values here — no need to touch other files for basic tuning.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "imagesoasis" / "Data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Dataset Balancing ────────────────────────────────────────────────────────
TRAIN_SAMPLES_PER_CLASS = 2000   # oversample/undersample to this number
TEST_SAMPLES_PER_CLASS = 800     # undersample majority test classes to this

# ─── Image ────────────────────────────────────────────────────────────────────
IMAGE_SIZE = (128, 128)           # (height, width)

# ─── Train / Val / Test split ─────────────────────────────────────────────────
TEST_SPLIT = 0.20                 # 20 % held out as test before anything else
VAL_SPLIT = 0.20                 # 20 % of the remaining train set → validation

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-5
RANDOM_SEED = 42

# ─── Model ────────────────────────────────────────────────────────────────────
DROPOUT_1 = 0.3
DROPOUT_2 = 0.5
DENSE_UNITS = 256
L2_REG = 1e-4

# ─── Early Stopping / LR Scheduler ───────────────────────────────────────────
ES_PATIENCE = 8
ES_MIN_DELTA = 1e-4
LR_FACTOR = 0.5
LR_PATIENCE = 3
LR_MIN = 1e-6

# ─── Checkpoint ───────────────────────────────────────────────────────────────
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"
FINAL_MODEL_PATH = MODELS_DIR / "alzheimer_model_final.pt"
