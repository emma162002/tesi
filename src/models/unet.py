"""
U-Net backbone for 3D pancreatic tumor segmentation.

Uses MONAI's highly optimized UNet implementation as the backbone.
Supports:
  - Standard 3-class segmentation (background, pancreas, tumor)
  - Feature extraction mode (returns encoder features for few-shot / MAE)
  - Loading pretrained encoder weights from self-supervised pretraining
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers import Norm


# ── Number of output classes ───────────────────────────────────────────────────
#   0 = background
#   1 = pancreas (parenchyma)
#   2 = tumor
NUM_CLASSES = 3


class PancreasUNet(nn.Module):
    """
    3D U-Net for pancreas + tumor segmentation.

    Architecture:
      - 5-level encoder-decoder with residual blocks
      - Instance Normalization (better than BN for small 3D batches)
      - Dropout for uncertainty estimation

    Args:
        in_channels:  number of input channels (1 for CT)
        num_classes:  number of output classes (default 3: bg / pancreas / tumor)
        features:     channel widths at each encoder level
        dropout:      dropout probability (0 = disabled)
        pretrained_encoder: path to encoder checkpoint from self-supervised pretraining
    """

    def __init__(
        self,
        in_channels:         int            = 1,
        num_classes:         int            = NUM_CLASSES,
        features:            Sequence[int]  = (32, 64, 128, 256, 512),
        dropout:             float          = 0.1,
        pretrained_encoder:  Optional[str]  = None,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.net = UNet(
            spatial_dims   = 3,
            in_channels    = in_channels,
            out_channels   = num_classes,
            channels       = features,
            strides        = (2, 2, 2, 2),
            num_res_units  = 2,
            norm           = Norm.INSTANCE,
            dropout        = dropout,
        )

        if pretrained_encoder is not None:
            self._load_pretrained_encoder(pretrained_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, num_classes, H, W, D)."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities of shape (B, num_classes, H, W, D)."""
        return torch.softmax(self.forward(x), dim=1)

    def _load_pretrained_encoder(self, path: str):
        """
        Load encoder weights from a self-supervised MAE checkpoint.
        Only keys matching the encoder (model.net.0 → model.net.model.0) are loaded.
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        # filter to encoder-side keys only
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.net.load_state_dict(encoder_state, strict=False)
        print(f"[PancreasUNet] Loaded pretrained encoder from {path}")
        print(f"  missing keys:    {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")


class SwinUNetR(nn.Module):
    """
    Swin-UNETR: transformer-based U-Net for 3D segmentation.
    Better than CNN U-Net on large datasets; heavier on memory.
    Use when GPU VRAM > 16 GB.

    Args:
        img_size:    input patch size (must be divisible by 32)
        in_channels: 1 for CT
        num_classes: 3 (bg / pancreas / tumor)
        feature_size: base feature dimension (48 or 96)
    """

    def __init__(
        self,
        img_size:     Tuple[int, int, int] = (96, 96, 96),
        in_channels:  int   = 1,
        num_classes:  int   = NUM_CLASSES,
        feature_size: int   = 48,
        pretrained:   Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = SwinUNETR(
            img_size     = img_size,
            in_channels  = in_channels,
            out_channels = num_classes,
            feature_size = feature_size,
            use_checkpoint = True,   # gradient checkpointing → saves VRAM
        )
        if pretrained is not None:
            ckpt = torch.load(pretrained, map_location="cpu")
            self.net.load_from(ckpt)
            print(f"[SwinUNetR] Loaded pretrained weights from {pretrained}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


def build_model(cfg: dict) -> nn.Module:
    """
    Factory function. Instantiates the right model from a config dict.

    Expected cfg keys:
      model_type:         'unet' | 'swinunetr'
      in_channels:        int
      num_classes:        int
      features:           list[int]   (unet only)
      dropout:            float
      pretrained_encoder: str | null
      img_size:           list[int]   (swinunetr only)
      feature_size:       int         (swinunetr only)
    """
    model_type = cfg.get("model_type", "unet").lower()

    if model_type == "unet":
        return PancreasUNet(
            in_channels        = cfg.get("in_channels", 1),
            num_classes        = cfg.get("num_classes", NUM_CLASSES),
            features           = cfg.get("features", [32, 64, 128, 256, 512]),
            dropout            = cfg.get("dropout", 0.1),
            pretrained_encoder = cfg.get("pretrained_encoder", None),
        )
    elif model_type == "swinunetr":
        return SwinUNetR(
            img_size     = tuple(cfg.get("img_size", [96, 96, 96])),
            in_channels  = cfg.get("in_channels", 1),
            num_classes  = cfg.get("num_classes", NUM_CLASSES),
            feature_size = cfg.get("feature_size", 48),
            pretrained   = cfg.get("pretrained_encoder", None),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'unet' or 'swinunetr'.")
