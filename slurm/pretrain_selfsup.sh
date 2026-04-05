#!/bin/bash
#SBATCH --job-name=pancreas_mae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/selfsup_%j.out
#SBATCH --error=logs/selfsup_%j.err

module load cuda/11.8
conda activate tesi

REPO="/home/$USER/tesi"
cd $REPO
mkdir -p logs experiments/selfsup_pretrain

python scripts/02_pretrain_selfsup.py \
    --config  configs/selfsup_pretrain.yaml \
    --output  experiments/selfsup_pretrain \
    --device  cuda

echo "Done: $?"
