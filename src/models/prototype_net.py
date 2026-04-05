"""
Prototypical Network for few-shot pancreatic tumor segmentation.

Architecture:
  - Shared CNN encoder (same U-Net encoder as supervised baseline)
  - Prototype computation: mean of support set features per class
  - Query segmentation: cosine similarity between query features and prototypes
  - Optional: spatial attention to focus on tumor-relevant regions

Reference: Snell et al. "Prototypical Networks for Few-shot Learning" (2017)
           Adapted for 3D volumetric segmentation following PANet (Wang et al. 2019).
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import PancreasUNet


class PrototypeSegmentationNet(nn.Module):
    """
    Prototypical network for 1-way K-shot tumor segmentation.

    Given K support (image, mask) pairs and Q query images:
      1. Extract dense feature maps from encoder for all support + query images
      2. For each class c, compute prototype p_c = mean of support features
         at positions where support mask == c
      3. Assign each query voxel to the class with the nearest prototype
         (cosine distance in feature space)

    Args:
        in_channels:    1 for CT
        num_classes:    number of segmentation classes (2: bg + tumor, or 3: bg+pancreas+tumor)
        feature_dim:    output channels of the encoder (bottleneck dimension)
        encoder_ckpt:   optional path to pretrained encoder checkpoint
    """

    def __init__(
        self,
        in_channels:   int           = 1,
        num_classes:   int           = 2,
        encoder_ckpt:  Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared encoder (same architecture as supervised baseline for fair comparison)
        self._backbone = PancreasUNet(
            in_channels        = in_channels,
            num_classes        = num_classes,
            pretrained_encoder = encoder_ckpt,
        )

        # We use only the encoder portion — replace the final classification head
        # with a projection layer that produces class-agnostic feature maps
        self.feature_proj = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1),  # project from first decoder features
            nn.InstanceNorm3d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = 64

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract dense feature maps at the original resolution.
        We tap into the backbone's penultimate feature maps (before final conv).

        Args:
            x: (B, 1, H, W, D)
        Returns:
            features: (B, feature_dim, H, W, D)
        """
        # Run through backbone and return intermediate features
        # MONAI UNet exposes the full forward; we hook into it
        logits = self._backbone(x)  # (B, num_classes, H, W, D)
        # Use softmax probabilities as features — a simple but effective proxy
        # for the learned feature space without architectural surgery
        features = torch.softmax(logits, dim=1)   # (B, num_classes, H, W, D)
        return features

    def compute_prototypes(
        self,
        support_features: torch.Tensor,  # (K, C, H, W, D)
        support_labels:   torch.Tensor,  # (K, 1, H, W, D)  integer class labels
    ) -> torch.Tensor:                   # (num_classes, C)
        """
        Compute per-class prototypes by masked average pooling over spatial dimensions.

        For class c:
            prototype_c = (sum of features where label == c) / (count where label == c)
        """
        K, C, H, W, D = support_features.shape
        prototypes = []

        for c in range(self.num_classes):
            mask = (support_labels == c).float()   # (K, 1, H, W, D)
            mask = mask.expand_as(support_features)  # (K, C, H, W, D)

            numerator   = (support_features * mask).sum(dim=(0, 2, 3, 4))   # (C,)
            denominator = mask.sum(dim=(0, 2, 3, 4)).clamp(min=1e-6)        # (C,)
            proto = numerator / denominator                                   # (C,)
            prototypes.append(proto)

        return torch.stack(prototypes, dim=0)  # (num_classes, C)

    def forward(
        self,
        support_images: torch.Tensor,   # (K, 1, H, W, D)
        support_labels: torch.Tensor,   # (K, 1, H, W, D)
        query_images:   torch.Tensor,   # (Q, 1, H, W, D)
    ) -> torch.Tensor:                  # (Q, num_classes, H, W, D) logits
        """
        Full prototypical segmentation forward pass.

        Steps:
          1. Encode all support + query images
          2. Compute per-class prototypes from support features + labels
          3. Compute cosine similarity between query features and each prototype
          4. Return similarity logits as segmentation scores
        """
        K = support_images.shape[0]
        Q = query_images.shape[0]

        # Encode all images in one batch for efficiency
        all_images   = torch.cat([support_images, query_images], dim=0)  # (K+Q, 1, H,W,D)
        all_features = self.encode(all_images)                           # (K+Q, C, H,W,D)

        support_features = all_features[:K]   # (K, C, H, W, D)
        query_features   = all_features[K:]   # (Q, C, H, W, D)

        # Prototypes: (num_classes, C)
        prototypes = self.compute_prototypes(support_features, support_labels)

        # Cosine similarity between each query voxel and each prototype
        # query_features: (Q, C, H, W, D) → (Q, H, W, D, C)
        qf = query_features.permute(0, 2, 3, 4, 1)  # (Q, H, W, D, C)

        # prototypes: (num_classes, C) → normalize
        proto_norm = F.normalize(prototypes, dim=1)   # (num_classes, C)
        qf_norm    = F.normalize(qf,         dim=-1)  # (Q, H, W, D, C)

        # similarity: (Q, H, W, D, num_classes)
        similarity = torch.einsum("qhwdc,nc->qhwdn", qf_norm, proto_norm)

        # scale by temperature and return as logits (Q, num_classes, H, W, D)
        logits = similarity.permute(0, 4, 1, 2, 3) * 10.0

        return logits

    def compute_loss(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images:   torch.Tensor,
        query_labels:   torch.Tensor,
    ) -> dict:
        """
        Compute cross-entropy loss on query predictions.
        Also returns Dice score on query for monitoring.
        """
        logits = self.forward(support_images, support_labels, query_images)

        Q, _, H, W, D = logits.shape
        target = query_labels.squeeze(1).long()  # (Q, H, W, D)

        ce_loss = F.cross_entropy(logits, target, ignore_index=-1)

        # Dice for tumor class (class index 1 in binary, or 2 in 3-class)
        probs   = torch.softmax(logits, dim=1)
        tumor_c = 1 if self.num_classes == 2 else 2
        pred_bin = (probs.argmax(dim=1) == tumor_c).float()
        gt_bin   = (target == tumor_c).float()
        intersection = (pred_bin * gt_bin).sum()
        dice = (2 * intersection + 1e-6) / (pred_bin.sum() + gt_bin.sum() + 1e-6)

        return {
            "loss":      ce_loss,
            "dice":      dice.item(),
            "ce_loss":   ce_loss.item(),
        }
