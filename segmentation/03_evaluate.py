"""
Evaluates segmentation quality by comparing predictions against ground truth.
Metrics: Dice score, volume error.
"""

import numpy as np
import nibabel as nib
from pathlib import Path


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int = 1) -> float:
    pred_bin = (pred == label).astype(np.uint8)
    gt_bin   = (gt   == label).astype(np.uint8)
    intersection = (pred_bin * gt_bin).sum()
    denominator  = pred_bin.sum() + gt_bin.sum()
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator


def volume_ml(mask: np.ndarray, voxel_spacing: tuple) -> float:
    """Volume in millilitres given voxel spacing in mm."""
    voxel_vol_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    return mask.sum() * voxel_vol_mm3 / 1000.0


def evaluate_case(pred_path: Path, gt_path: Path, label: int = 1) -> dict:
    pred_nii = nib.load(pred_path)
    gt_nii   = nib.load(gt_path)

    pred    = np.array(pred_nii.dataobj)
    gt      = np.array(gt_nii.dataobj)
    spacing = pred_nii.header.get_zooms()

    dice     = dice_score(pred, gt, label)
    vol_pred = volume_ml((pred == label), spacing)
    vol_gt   = volume_ml((gt   == label), spacing)

    return {
        "case":          pred_path.stem,
        "dice":          round(dice, 4),
        "vol_pred_ml":   round(vol_pred, 2),
        "vol_gt_ml":     round(vol_gt, 2),
        "vol_error_ml":  round(abs(vol_pred - vol_gt), 2),
    }


def evaluate_folder(pred_dir: Path, gt_dir: Path, label: int = 1) -> list:
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    results = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  [SKIP] ground truth not found for {pred_path.name}")
            continue
        r = evaluate_case(pred_path, gt_path, label)
        results.append(r)
        print(f"  {r['case']:30s}  Dice: {r['dice']:.4f}  "
              f"Vol pred: {r['vol_pred_ml']:6.1f} ml  Vol GT: {r['vol_gt_ml']:6.1f} ml")

    if results:
        mean_dice = np.mean([r["dice"] for r in results])
        print(f"\nMean Dice: {mean_dice:.4f} over {len(results)} cases")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python 03_evaluate.py <pred_dir> <gt_dir>")
        sys.exit(1)

    pred_dir = Path(sys.argv[1])
    gt_dir   = Path(sys.argv[2])
    evaluate_folder(pred_dir, gt_dir, label=1)
