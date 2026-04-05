"""Utilities for loading/saving NIfTI files and checkpoints."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn


def load_nifti(path: str) -> Tuple[np.ndarray, tuple, np.ndarray]:
    """Load a NIfTI file. Returns (array, voxel_spacing_mm, affine)."""
    nii = nib.load(path)
    return np.asarray(nii.dataobj), nii.header.get_zooms()[:3], nii.affine


def save_nifti(array: np.ndarray, affine: np.ndarray, path: str):
    """Save a numpy array as a NIfTI file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.int16), affine), path)


def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    path:      str,
    scheduler=None,
):
    """Save a full training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "metrics":          metrics,
    }
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device:    str = "cpu",
) -> Tuple[int, dict]:
    """Load a checkpoint. Returns (epoch, metrics)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']} from {path}")
    return ckpt["epoch"], ckpt.get("metrics", {})


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def save_results_json(results: dict, path: str):
    """Save evaluation results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved → {path}")
