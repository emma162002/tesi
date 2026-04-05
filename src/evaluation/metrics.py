"""
Evaluation metrics for 3D pancreatic segmentation.

Metrics implemented:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU / Jaccard)
  - 95th percentile Hausdorff Distance (HD95)
  - Average Symmetric Surface Distance (ASSD)
  - Volume Error (ml)
  - Precision, Recall (sensitivity/specificity)

All metrics are computed per-case, per-class, then averaged.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch
from scipy.ndimage import binary_erosion, label as scipy_label
from scipy.spatial.distance import directed_hausdorff


# ── Voxel-level metrics ────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice Similarity Coefficient for binary masks.
    Returns 1.0 if both masks are empty (vacuously correct).
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    intersection = (pred & gt).sum()
    denominator  = pred.sum() + gt.sum()

    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union (Jaccard index) for binary masks."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Precision and recall for binary masks."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    return precision, recall


# ── Surface-based metrics ──────────────────────────────────────────────────────

def _get_surface_points(mask: np.ndarray, spacing: tuple) -> np.ndarray:
    """
    Extract surface voxel coordinates from a binary mask using erosion.
    Accounts for anisotropic voxel spacing.

    Returns array of shape (N, 3) with physical coordinates in mm.
    """
    eroded  = binary_erosion(mask)
    surface = mask & ~eroded
    coords  = np.argwhere(surface).astype(float)

    # scale by voxel spacing to get mm coordinates
    coords *= np.array(spacing)
    return coords


def hausdorff_distance_95(
    pred:    np.ndarray,
    gt:      np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """
    95th percentile Hausdorff Distance in mm.
    Robust to outliers compared to max HD.
    Returns 0 if both masks are empty, inf if one is empty.
    """
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return float("inf")

    pred_pts = _get_surface_points(pred_b, spacing)
    gt_pts   = _get_surface_points(gt_b,   spacing)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    # directed distances
    from scipy.spatial import cKDTree
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    hd95 = max(
        np.percentile(d_pred_to_gt, 95),
        np.percentile(d_gt_to_pred, 95),
    )
    return float(hd95)


def average_surface_distance(
    pred:    np.ndarray,
    gt:      np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """Average Symmetric Surface Distance (ASSD) in mm."""
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)

    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return float("inf")

    from scipy.spatial import cKDTree
    pred_pts = _get_surface_points(pred_b, spacing)
    gt_pts   = _get_surface_points(gt_b,   spacing)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf")

    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d1, _ = tree_gt.query(pred_pts)
    d2, _ = tree_pred.query(gt_pts)

    return float((d1.mean() + d2.mean()) / 2)


# ── Volume metrics ─────────────────────────────────────────────────────────────

def volume_ml(mask: np.ndarray, spacing: tuple) -> float:
    """Compute volume in millilitres given binary mask and voxel spacing (mm)."""
    voxel_vol_mm3 = float(np.prod(spacing))
    return float(mask.astype(bool).sum() * voxel_vol_mm3 / 1000.0)


# ── Per-case evaluation ────────────────────────────────────────────────────────

def evaluate_case(
    pred:    np.ndarray,       # (H, W, D) integer labels
    gt:      np.ndarray,       # (H, W, D) integer labels
    spacing: tuple = (1.0, 1.0, 1.0),
    classes: dict  = {1: "pancreas", 2: "tumor"},
    compute_surface_metrics: bool = True,
) -> Dict[str, float]:
    """
    Full per-case evaluation returning all metrics for each class.

    Args:
        pred:     predicted segmentation (integer class labels)
        gt:       ground truth segmentation
        spacing:  voxel spacing in mm (x, y, z)
        classes:  dict mapping class index to class name
        compute_surface_metrics: HD95 and ASSD are slow — set False for quick eval

    Returns:
        flat dict with keys like 'pancreas_dice', 'tumor_hd95', etc.
    """
    results = {}

    for cls_idx, cls_name in classes.items():
        pred_bin = (pred == cls_idx)
        gt_bin   = (gt   == cls_idx)

        results[f"{cls_name}_dice"]     = dice_score(pred_bin, gt_bin)
        results[f"{cls_name}_iou"]      = iou_score(pred_bin, gt_bin)
        prec, rec = precision_recall(pred_bin, gt_bin)
        results[f"{cls_name}_precision"] = prec
        results[f"{cls_name}_recall"]    = rec
        results[f"{cls_name}_vol_pred"]  = volume_ml(pred_bin, spacing)
        results[f"{cls_name}_vol_gt"]    = volume_ml(gt_bin,   spacing)
        results[f"{cls_name}_vol_error"] = abs(
            results[f"{cls_name}_vol_pred"] - results[f"{cls_name}_vol_gt"]
        )

        if compute_surface_metrics:
            results[f"{cls_name}_hd95"] = hausdorff_distance_95(pred_bin, gt_bin, spacing)
            results[f"{cls_name}_assd"] = average_surface_distance(pred_bin, gt_bin, spacing)

    return results


def aggregate_metrics(per_case_results: list) -> Dict[str, float]:
    """
    Compute mean ± std across all cases for each metric.
    Infinite HD95 values (empty predictions) are excluded from mean.

    Returns dict with keys like 'tumor_dice_mean', 'tumor_dice_std', etc.
    """
    if not per_case_results:
        return {}

    all_keys = per_case_results[0].keys()
    agg = {}

    for key in all_keys:
        values = [r[key] for r in per_case_results if np.isfinite(r[key])]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"]  = float(np.std(values))
            agg[f"{key}_n"]    = len(values)

    return agg
