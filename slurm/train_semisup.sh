#!/bin/bash
#SBATCH --job-name=pancreas_semisup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/semisup_%j.out
#SBATCH --error=logs/semisup_%j.err

# ── Environment setup ──────────────────────────────────────────────────────────
module load cuda/11.8
conda activate tesi   # or: source ~/miniconda3/bin/activate tesi

REPO="/home/$USER/tesi"
cd $REPO

mkdir -p logs experiments/semisup

# ── Run training ───────────────────────────────────────────────────────────────
python scripts/03_train_semisup.py \
    --config   configs/semisup.yaml \
    --output   experiments/semisup \
    --device   cuda

echo "Done: $?"
