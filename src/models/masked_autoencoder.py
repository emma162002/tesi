"""
3D Masked Autoencoder (MAE) for self-supervised pretraining on CT volumes.

Architecture:
  - Encoder: 3D CNN (U-Net encoder half) that processes visible patches
  - Decoder: lightweight 3D CNN that reconstructs masked patches
  - Masking:  random cubic patch masking (tube masking adapted for 3D)

Reference: He et al. "Masked Autoencoders Are Scalable Vision Learners" (2021)
           Adapted for volumetric CT by replacing ViT patches with 3D conv patches.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.networks.layers import Norm


class PatchMasker3D(nn.Module):
    """
    Randomly masks cubic patches of a 3D volume.

    Strategy:
      1. Divide the volume into non-overlapping patches of size patch_size^3
      2. Randomly select mask_ratio fraction of patches to mask
      3. Replace masked patches with a learnable [MASK] token

    Args:
        patch_size:  size of each cubic patch (e.g. 16 → 16^3 voxels)
        mask_ratio:  fraction of patches to mask (e.g. 0.75)
    """

    def __init__(self, patch_size: int = 16, mask_ratio: float = 0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W, D)
        Returns:
            masked_x:  (B, C, H, W, D) — original with masked regions replaced
            mask:      (B, H, W, D) bool tensor — True where masked
        """
        B, C, H, W, D = x.shape
        ps = self.patch_size

        # number of patches per dimension
        nH, nW, nD = H // ps, W // ps, D // ps
        n_patches   = nH * nW * nD
        n_masked    = int(n_patches * self.mask_ratio)

        # generate random patch indices to mask, per batch element
        mask = torch.zeros(B, nH, nW, nD, dtype=torch.bool, device=x.device)
        for b in range(B):
            idx = torch.randperm(n_patches, device=x.device)[:n_masked]
            # convert flat idx → 3D patch indices
            i = idx // (nW * nD)
            j = (idx % (nW * nD)) // nD
            k = idx % nD
            mask[b, i, j, k] = True

        # upsample patch mask to voxel mask
        voxel_mask = mask.unsqueeze(1).float()                        # (B,1,nH,nW,nD)
        voxel_mask = F.interpolate(voxel_mask,
                                   size=(H, W, D),
                                   mode="nearest").squeeze(1).bool()  # (B,H,W,D)

        # apply mask token
        masked_x = x.clone()
        masked_x[voxel_mask.unsqueeze(1).expand_as(x)] = \
            self.mask_token.expand_as(x)[voxel_mask.unsqueeze(1).expand_as(x)]

        return masked_x, voxel_mask


class MAEEncoder(nn.Module):
    """
    Encoder half of the MAE: a 3D CNN that maps CT volumes to latent features.
    Reuses the U-Net encoder architecture for compatibility with downstream fine-tuning.
    """

    def __init__(self, in_channels: int = 1,
                 features: Tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()

        # We reuse the full UNet but only use encoder side during pretraining.
        # During fine-tuning, the entire UNet (encoder + decoder) is fine-tuned.
        self._features = features

        self.encoder = nn.Sequential(
            # Level 1: 1 → 32
            self._conv_block(in_channels, features[0]),
            # Level 2: 32 → 64 (stride 2)
            self._conv_block(features[0], features[1], stride=2),
            # Level 3: 64 → 128
            self._conv_block(features[1], features[2], stride=2),
            # Level 4: 128 → 256
            self._conv_block(features[2], features[3], stride=2),
            # Bottleneck: 256 → 512
            self._conv_block(features[3], features[4], stride=2),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MAEDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs the original CT volume from latent features.
    Uses transposed convolutions to upsample back to input resolution.
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 1,
                 features: Tuple[int, ...] = (256, 128, 64, 32)):
        super().__init__()

        layers = []
        ch = in_channels
        for f in features:
            layers += [
                nn.ConvTranspose3d(ch, f, kernel_size=2, stride=2),
                nn.InstanceNorm3d(f, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            ch = f

        layers += [nn.Conv3d(ch, out_channels, kernel_size=1)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class MaskedAutoencoder3D(nn.Module):
    """
    Full MAE model: masker + encoder + decoder.

    Training objective: reconstruct the original CT intensity values
    at the masked voxel positions (MSE loss on masked voxels only).

    Args:
        in_channels:   1 (CT is single-channel)
        enc_features:  encoder channel widths
        dec_features:  decoder channel widths (reversed encoder)
        patch_size:    masking patch size in voxels
        mask_ratio:    fraction of patches to mask during pretraining
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        enc_features: Tuple  = (32, 64, 128, 256, 512),
        dec_features: Tuple  = (256, 128, 64, 32),
        patch_size:   int   = 16,
        mask_ratio:   float = 0.75,
    ):
        super().__init__()
        self.masker  = PatchMasker3D(patch_size=patch_size, mask_ratio=mask_ratio)
        self.encoder = MAEEncoder(in_channels=in_channels, features=enc_features)
        self.decoder = MAEDecoder(in_channels=enc_features[-1],
                                   out_channels=in_channels,
                                   features=dec_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  (B, 1, H, W, D) — original CT patch
        Returns:
            recon:      (B, 1, H, W, D) — reconstructed volume
            mask:       (B, H, W, D)   — True at masked positions
            latent:     (B, C, h, w, d) — bottleneck features (for downstream use)
        """
        masked_x, mask = self.masker(x)
        latent = self.encoder(masked_x)
        recon  = self.decoder(latent)
        return recon, mask, latent

    def compute_loss(self, x: torch.Tensor) -> dict:
        """
        Compute masked reconstruction loss.
        Loss is computed ONLY at masked positions → forces model to use context.

        Returns dict with 'loss', 'recon_loss' for logging.
        """
        recon, mask, _ = self.forward(x)

        # resize recon to match x if necessary
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:], mode="trilinear",
                                  align_corners=False)

        # MSE only on masked voxels
        mask_expanded = mask.unsqueeze(1).expand_as(x)  # (B,1,H,W,D)
        if mask_expanded.sum() == 0:
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        else:
            loss = F.mse_loss(recon[mask_expanded], x[mask_expanded])

        return {"loss": loss, "recon_loss": loss.item()}

    def get_encoder(self) -> MAEEncoder:
        """Return encoder for downstream fine-tuning."""
        return self.encoder
