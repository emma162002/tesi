"""
Partial / masked cross-entropy loss for weakly supervised segmentation.

Weak annotation types supported:
  - Scribble annotations:   label = 1 (fg scribble), 2 (bg scribble), 0 (unknown)
  - Bounding-box labels:    label = 1 (inside box), 0 (outside — treated as bg)
  - Presence/absence:       volume-level binary flag (expands to full-volume label)

The key idea: compute cross-entropy ONLY at annotated voxels (where label != IGNORE).
Unknown voxels are masked out from the gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


IGNORE_INDEX = -1   # voxels with this label value are skipped in the loss


def remap_scribble_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Convert scribble label convention to standard convention:
      0 (unknown) → IGNORE_INDEX (-1)
      1 (fg/tumor scribble) → 1
      2 (bg scribble) → 0
    """
    out = labels.clone().long()
    out[labels == 0] = IGNORE_INDEX
    out[labels == 1] = 1
    out[labels == 2] = 0
    return out


class PartialCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss that ignores unlabeled (unknown) voxels.

    Args:
        ignore_index:  voxel value to ignore (default -1)
        label_type:    'scribble' | 'bbox' | 'full'
                       'scribble' → remaps 0 to IGNORE before computing loss
                       'bbox'     → uses raw labels (0=outside, 1=inside)
                       'full'     → standard CE (all voxels)
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX, label_type: str = "scribble"):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_type   = label_type
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,   # (B, C, H, W, D)
        labels: torch.Tensor,   # (B, 1, H, W, D) integer
    ) -> dict:
        target = labels.squeeze(1).long()   # (B, H, W, D)

        if self.label_type == "scribble":
            target = remap_scribble_labels(target)

        loss = self.ce(logits, target)

        # fraction of annotated voxels (for monitoring annotation density)
        annotated_frac = (target != self.ignore_index).float().mean().item()

        return {
            "loss":           loss,
            "ce_loss":        loss.item(),
            "annotated_frac": annotated_frac,
        }


class WeakSupLoss(nn.Module):
    """
    Combined loss for weakly supervised training.

    Components:
      1. Partial CE on weak labels (at annotated voxels only)
      2. Regularization: size constraint loss to prevent trivial solutions
         (model should not predict empty or full-volume segmentations)

    Args:
        label_type:       'scribble' | 'bbox'
        size_lambda:      weight for size regularization term
        expected_size:    expected tumor fraction (e.g. 0.05 = 5% of patch)
    """

    def __init__(
        self,
        label_type:    str   = "scribble",
        size_lambda:   float = 0.1,
        expected_size: float = 0.05,
    ):
        super().__init__()
        self.partial_ce    = PartialCrossEntropyLoss(label_type=label_type)
        self.size_lambda   = size_lambda
        self.expected_size = expected_size

    def size_constraint_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Penalizes predicted segmentation sizes that deviate from expected_size.
        Prevents degenerate solutions (all-bg or all-fg predictions).
        """
        probs = torch.softmax(logits, dim=1)            # (B, C, H, W, D)
        fg_prob = probs[:, 1:].sum(dim=1)               # (B, H, W, D) foreground prob

        # Mean predicted foreground fraction per batch element
        pred_size = fg_prob.mean(dim=(1, 2, 3))         # (B,)
        target_size = torch.full_like(pred_size, self.expected_size)

        return F.mse_loss(pred_size, target_size)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        ce_result  = self.partial_ce(logits, labels)
        size_loss  = self.size_constraint_loss(logits)

        total = ce_result["loss"] + self.size_lambda * size_loss

        return {
            "loss":           total,
            "ce_loss":        ce_result["ce_loss"],
            "size_loss":      size_loss.item(),
            "annotated_frac": ce_result["annotated_frac"],
        }
