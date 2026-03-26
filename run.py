"""
run.py  —  project root launcher
Usage:
    python run.py train
    python run.py evaluate
"""

import sys
from pathlib import Path

# Make sure the project root is always on sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "evaluate"):
    print("Usage:  python run.py train | evaluate")
    sys.exit(1)

if sys.argv[1] == "train":
    from src.train import train
    train()
else:
    from src.evaluate import evaluate
    evaluate()
