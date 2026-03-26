"""
utils/device.py
Selects the best available device: MPS (Apple Silicon) → CUDA → CPU.
"""

import torch


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Priority:
      1. MPS  — Apple Silicon (M1/M2/M3/M4) GPUs
      2. CUDA — NVIDIA GPU (not available on Mac)
      3. CPU  — fallback
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] Using CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] Using CPU")
    return device
