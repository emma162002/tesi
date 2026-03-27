"""
nnUNet training script for pancreas segmentation.
Intended to be run on the university GPU server.

Commands to run in sequence (terminal):

1. Preprocessing:
   nnUNetv2_plan_and_preprocess -d 7 --verify_dataset_integrity

2. Training (3 folds for cross-validation):
   nnUNetv2_train 7 3d_fullres 0 --npz
   nnUNetv2_train 7 3d_fullres 1 --npz
   nnUNetv2_train 7 3d_fullres 2 --npz

3. Find best configuration:
   nnUNetv2_find_best_configuration 7 -c 3d_fullres

4. Inference on new cases:
   nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 7 -c 3d_fullres --save_probabilities
"""

import subprocess
import sys

DATASET_ID = 7
CONFIG = "3d_fullres"
FOLDS = [0, 1, 2]


def run_preprocessing():
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(DATASET_ID),
        "--verify_dataset_integrity"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_training(fold: int):
    cmd = [
        "nnUNetv2_train",
        str(DATASET_ID), CONFIG, str(fold),
        "--npz"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_inference(input_dir: str, output_dir: str):
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", str(DATASET_ID),
        "-c", CONFIG,
        "--save_probabilities"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_train.py [preprocess|train|inference]")
        print("  preprocess            -> prepare data")
        print("  train                 -> run training on all folds")
        print("  inference <in> <out>  -> run inference on new data")
        sys.exit(1)

    action = sys.argv[1]

    if action == "preprocess":
        run_preprocessing()
    elif action == "train":
        for fold in FOLDS:
            print(f"\n=== Training fold {fold} ===")
            run_training(fold)
    elif action == "inference":
        if len(sys.argv) < 4:
            print("Usage: python 02_train.py inference <input_dir> <output_dir>")
            sys.exit(1)
        run_inference(sys.argv[2], sys.argv[3])
