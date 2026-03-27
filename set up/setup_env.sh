#!/bin/bash
# Full environment setup for the thesis project

echo "=== Creating conda environment ==="
conda env create -f environment.yml

echo ""
echo "=== Activate the environment with: ==="
echo "conda activate tesi"
echo ""
echo "=== Then add these lines to your ~/.zshrc: ==="
echo 'export nnUNet_raw="/Users/emmabattaglia/tesi/data/nnunet_raw"'
echo 'export nnUNet_preprocessed="/Users/emmabattaglia/tesi/data/nnunet_preprocessed"'
echo 'export nnUNet_results="/Users/emmabattaglia/tesi/data/nnunet_results"'
