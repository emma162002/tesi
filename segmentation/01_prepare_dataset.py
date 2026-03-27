"""
Converts the Medical Segmentation Decathlon dataset (Task07_Pancreas)
into the format required by nnUNet.

Expected input structure:
    data/Task07_Pancreas/
        imagesTr/   -> CT training images (*.nii.gz)
        labelsTr/   -> training segmentations (*.nii.gz)
        imagesTs/   -> CT test images (*.nii.gz)
        dataset.json

Output structure (nnUNet):
    data/nnunet_raw/Dataset007_Pancreas/
        imagesTr/   -> pancreas_001_0000.nii.gz, ...
        labelsTr/   -> pancreas_001.nii.gz, ...
        dataset.json
"""

import os
import shutil
import json
from pathlib import Path

RAW_DATASET  = Path("/Users/emmabattaglia/tesi/data/Task07_Pancreas")
NNUNET_RAW   = Path(os.environ.get("nnUNet_raw", "/Users/emmabattaglia/tesi/data/nnunet_raw"))
DATASET_ID   = 7
DATASET_NAME = f"Dataset{DATASET_ID:03d}_Pancreas"
OUTPUT_DIR   = NNUNET_RAW / DATASET_NAME


def convert_dataset():
    print(f"Input:  {RAW_DATASET}")
    print(f"Output: {OUTPUT_DIR}")

    if not RAW_DATASET.exists():
        print("\nERROR: dataset not found.")
        print(f"Download it from medicaldecathlon.com and place it at: {RAW_DATASET}")
        return

    (OUTPUT_DIR / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labelsTr").mkdir(parents=True, exist_ok=True)

    images = sorted((RAW_DATASET / "imagesTr").glob("*.nii.gz"))
    labels = sorted((RAW_DATASET / "labelsTr").glob("*.nii.gz"))

    print(f"\nFound {len(images)} images, {len(labels)} labels")

    for i, (img, lbl) in enumerate(zip(images, labels), start=1):
        case_id = f"pancreas_{i:03d}"
        shutil.copy(img, OUTPUT_DIR / "imagesTr" / f"{case_id}_0000.nii.gz")
        shutil.copy(lbl, OUTPUT_DIR / "labelsTr" / f"{case_id}.nii.gz")
        print(f"  [{i:3d}/{len(images)}] {img.name} -> {case_id}")

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "pancreas": 1,
            "cancer": 2
        },
        "numTraining": len(images),
        "file_ending": ".nii.gz"
    }
    with open(OUTPUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset ready at: {OUTPUT_DIR}")
    print("\nNext step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")


if __name__ == "__main__":
    convert_dataset()
