"""
Standard Dice + Cross-Entropy loss for fully supervised segmentation.
Used as the baseline loss for the supervised trainer and as the supervised
component inside semi-supervised training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, FocalLoss


class DiceCECombined(nn.Module):
    """
    Weighted combination of Dice loss and Cross-Entropy loss.

    Dice loss handles class imbalance well (important for small tumors).
    CE loss provides stable gradients early in training.

    Args:
        num_classes:    number of segmentation classes
        dice_weight:    weight for Dice loss component
        ce_weight:      weight for CE loss component
        class_weights:  per-class CE weights (e.g. [1.0, 1.0, 5.0] to upweight tumor)
        smooth_dr:      Dice denominator smoothing factor
        smooth_nr:      Dice numerator smoothing factor
    """

    def __init__(
        self,
        num_classes:   int   = 3,
        dice_weight:   float = 0.5,
        ce_weight:     float = 0.5,
        class_weights: list  = None,
        smooth_dr:     float = 1e-6,
        smooth_nr:     float = 0.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.num_classes = num_classes

        self.dice_loss = DiceLoss(
            include_background = False,   # skip background class in Dice
            to_onehot_y       = True,
            softmax           = True,
            smooth_dr         = smooth_dr,
            smooth_nr         = smooth_nr,
        )

        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(
        self,
        logits: torch.Tensor,   # (B, C, H, W, D) — raw model output
        labels: torch.Tensor,   # (B, 1, H, W, D) — integer class indices
    ) -> dict:
        """
        Returns dict with 'loss', 'dice_loss', 'ce_loss' for logging.
        """
        # Dice loss expects (B, C, H, W, D) logits and (B, 1, H, W, D) integer labels
        d_loss = self.dice_loss(logits, labels)

        # CE loss expects (B, C, H, W, D) logits and (B, H, W, D) integer labels
        c_loss = self.ce_loss(logits, labels.squeeze(1).long())

        total = self.dice_weight * d_loss + self.ce_weight * c_loss

        return {
            "loss":      total,
            "dice_loss": d_loss.item(),
            "ce_loss":   c_loss.item(),
        }


class TumorFocusedLoss(nn.Module):
    """
    Variant of DiceCE that applies extra focus on the tumor class (class 2).
    Uses Focal loss for the CE component to handle extreme class imbalance
    in small pancreatic tumors (tumor occupies < 0.1% of CT volume).

    Args:
        num_classes:    3 (bg / pancreas / tumor)
        focal_gamma:    focal loss gamma parameter (higher → more focus on hard voxels)
        tumor_weight:   extra weight multiplier for tumor class in Dice
    """

    def __init__(
        self,
        num_classes:  int   = 3,
        focal_gamma:  float = 2.0,
        tumor_weight: float = 3.0,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.tumor_weight = tumor_weight

        self.dice_loss = DiceLoss(
            include_background = False,
            to_onehot_y       = True,
            softmax           = True,
        )
        self.focal_loss = FocalLoss(
            include_background = False,
            to_onehot_y       = True,
            gamma             = focal_gamma,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        d_loss = self.dice_loss(logits, labels)
        f_loss = self.focal_loss(logits, labels)

        # Extra penalty on tumor Dice specifically
        probs    = torch.softmax(logits, dim=1)  # (B, C, H, W, D)
        gt_tumor = (labels == 2).float()         # (B, 1, H, W, D)
        pred_t   = probs[:, 2:3]                 # (B, 1, H, W, D)
        inter    = (pred_t * gt_tumor).sum()
        tumor_dice_loss = 1.0 - (2 * inter + 1e-6) / (
            pred_t.sum() + gt_tumor.sum() + 1e-6
        )

        total = d_loss + f_loss + self.tumor_weight * tumor_dice_loss

        return {
            "loss":            total,
            "dice_loss":       d_loss.item(),
            "focal_loss":      f_loss.item(),
            "tumor_dice_loss": tumor_dice_loss.item(),
        }


def build_loss(cfg: dict) -> nn.Module:
    """Factory for loss functions."""
    loss_type = cfg.get("loss_type", "dice_ce")
    if loss_type == "dice_ce":
        return DiceCECombined(
            num_classes   = cfg.get("num_classes", 3),
            dice_weight   = cfg.get("dice_weight", 0.5),
            ce_weight     = cfg.get("ce_weight", 0.5),
            class_weights = cfg.get("class_weights", None),
        )
    elif loss_type == "tumor_focused":
        return TumorFocusedLoss(
            num_classes  = cfg.get("num_classes", 3),
            focal_gamma  = cfg.get("focal_gamma", 2.0),
            tumor_weight = cfg.get("tumor_weight", 3.0),
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
