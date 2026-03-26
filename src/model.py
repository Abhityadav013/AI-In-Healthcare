"""
src/model.py
EfficientNetB0 transfer learning model in PyTorch.

The original notebook used TensorFlow/Keras with a manually downloaded weight
file from Kaggle.  Here we use the `timm` library (PyTorch Image Models) which
downloads pretrained ImageNet weights automatically and works perfectly on MPS.

Why switch from TF/Keras to PyTorch + timm?
  - Full MPS support on Apple Silicon (TF MPS support is partial and buggy).
  - timm has EfficientNetB0 with pretrained weights ready out of the box.
  - No need to download .h5 weight files manually.
  - PyTorch is now the dominant framework in ML research.

Architecture mirrors the notebook:
  EfficientNetB0 (pretrained, fully fine-tuned)
  → BatchNorm
  → GlobalAveragePooling (done inside EfficientNetB0 head)
  → Dropout(0.3)
  → Dense(512, relu) + L2 weight decay (via optimizer)
  → Dropout(0.5)
  → Dense(4, softmax)  ← actually linear; softmax is in the loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "The `timm` package is required.  Install it with:  pip install timm"
    )

from config import DENSE_UNITS, DROPOUT_1, DROPOUT_2, NUM_CLASSES


class AlzheimerClassifier(nn.Module):
    """
    EfficientNetB0-based classifier for Alzheimer's MRI severity.

    Args:
        num_classes:   Number of output classes (default: 4).
        pretrained:    Load ImageNet pretrained weights (default: True).
        dropout1:      Dropout rate before the dense layer.
        dropout2:      Dropout rate after the dense layer.
        dense_units:   Size of the intermediate dense layer.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout1: float = DROPOUT_1,
        dropout2: float = DROPOUT_2,
        dense_units: int = DENSE_UNITS,
    ) -> None:
        super().__init__()

        # Load EfficientNetB0 backbone — pretrained on ImageNet
        # num_classes=0 removes the default classifier head so we get raw features
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,      # strip default head → outputs (B, 1280)
            global_pool="avg",  # GlobalAveragePooling built-in
        )

        # Get the feature dimension from the backbone
        feature_dim = self.backbone.num_features  # 1280 for EfficientNetB0

        # Custom classification head (mirrors the Keras notebook)
        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout1),
            nn.Linear(feature_dim, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(dense_units, num_classes),
            # No softmax here — CrossEntropyLoss includes it internally.
            # For inference we apply softmax manually (see evaluate.py).
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 1280)
        logits = self.head(features)  # (B, num_classes)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone weights (useful for a warm-up phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[model] Backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone weights for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[model] Backbone unfrozen — full fine-tuning enabled.")

    def count_parameters(self) -> tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel()
                        for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> AlzheimerClassifier:
    """
    Factory function to create and summarise the model.

    Args:
        num_classes: Number of output classes.
        pretrained:  Use ImageNet pretrained weights.

    Returns:
        AlzheimerClassifier instance (on CPU — move to device in train.py).
    """
    model = AlzheimerClassifier(num_classes=num_classes, pretrained=pretrained)

    total, trainable = model.count_parameters()
    print(f"\n[model] AlzheimerClassifier (EfficientNetB0)")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model
