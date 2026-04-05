# Pancreatic Tumor Segmentation — Master Thesis
**Sugimoto Lab | Biomedical Engineering**

Research Topic 3: *Pancreatic Tumor Segmentation with Low Estimation Accuracy and Limited Annotations*

---

## Overview

This project implements and compares three annotation-efficient deep learning strategies for pancreatic tumor segmentation on CT images, addressing the dual challenge of low segmentation accuracy and annotation scarcity:

1. **Semi-supervised learning** — Mean Teacher with pseudo-label generation (small labeled set + large unlabeled set)
2. **Self-supervised pretraining** — 3D Masked Autoencoder (MAE) pretrained on unlabeled CT, then fine-tuned
3. **Few-shot segmentation** — Prototypical Networks for rapid adaptation from K annotated examples

All methods are evaluated against a fully supervised baseline using Dice, IoU, HD95, and volume error on the Medical Segmentation Decathlon Task07 (pancreas + tumor).

---

## Project Structure

```
tesi/
├── configs/                        ← YAML config file per experiment
│   ├── supervised_baseline.yaml
│   ├── semisup.yaml
│   ├── selfsup_pretrain.yaml
│   └── fewshot.yaml
│
├── src/                            ← Core importable library
│   ├── data/
│   │   ├── dataset.py              ← Supervised, semi-sup, weak-label, few-shot datasets
│   │   └── transforms.py          ← 3D CT augmentation pipelines (MONAI)
│   ├── models/
│   │   ├── unet.py                 ← 3D U-Net and Swin-UNETR backbone
│   │   ├── masked_autoencoder.py   ← 3D MAE for self-supervised pretraining
│   │   └── prototype_net.py        ← Prototypical network for few-shot segmentation
│   ├── losses/
│   │   ├── dice_ce.py              ← Dice + CE and tumor-focused focal loss
│   │   ├── partial_ce.py          ← Masked CE for weak/scribble annotations
│   │   └── consistency.py         ← Mean Teacher EMA + pseudo-label loss
│   ├── training/
│   │   ├── trainer_supervised.py   ← Baseline fully supervised trainer
│   │   ├── trainer_semisup.py      ← Semi-supervised Mean Teacher trainer
│   │   ├── trainer_selfsup.py      ← MAE pretrainer + fine-tuner
│   │   └── trainer_fewshot.py      ← Episodic few-shot trainer
│   ├── evaluation/
│   │   └── metrics.py              ← Dice, IoU, HD95, ASSD, volume error
│   └── utils/
│       └── io.py                   ← NIfTI I/O, checkpoint save/load, config loader
│
├── scripts/                        ← CLI entry points (for SLURM cluster)
│   ├── 02_pretrain_selfsup.py
│   └── 03_train_semisup.py
│
├── notebooks/
│   └── 00_colab_train.ipynb        ← All-in-one Colab training notebook
│
├── slurm/                          ← SLURM job scripts for university GPU cluster
│   ├── pretrain_selfsup.sh
│   └── train_semisup.sh
│
├── segmentation/                   ← Legacy nnUNet pipeline (kept as reference)
├── llm/                            ← LLM report generation (Ollama / Claude API)
├── pipeline/                       ← End-to-end pipeline
├── data/                           ← Ignored by git (too large)
│   ├── Task07_Pancreas/
│   ├── nnunet_raw/
│   └── unlabeled/
└── experiments/                    ← Ignored by git (checkpoints, logs)
```

---

## Setup

### Option A — Google Colab (recommended for development)

Open `notebooks/00_colab_train.ipynb` in Colab, mount your Drive, and run. The notebook handles installation automatically.

```
Runtime → Change runtime type → A100 GPU (Colab Pro)
```

### Option B — University GPU cluster (recommended for full training runs)

```bash
# 1. Clone the repo on the cluster
git clone https://github.com/emma162002/tesi.git
cd tesi

# 2. Create the conda environment
conda env create -f "set up/environment.yml"
conda activate tesi

# 3. Set nnUNet environment variables (add to ~/.bashrc)
export nnUNet_raw="$HOME/tesi/data/nnunet_raw"
export nnUNet_preprocessed="$HOME/tesi/data/nnunet_preprocessed"
export nnUNet_results="$HOME/tesi/data/nnunet_results"

# 4. Submit a job
sbatch slurm/train_semisup.sh
```

### Option C — Local machine (debugging only)

```bash
conda env create -f "set up/environment.yml"
conda activate tesi
python scripts/03_train_semisup.py --config configs/semisup.yaml --output experiments/semisup
```

---

## Workflow

| Step | Where | How |
|------|-------|-----|
| 1. Download dataset | Local | medicaldecathlon.com → Task07_Pancreas |
| 2. Convert to nnUNet format | Local | `python segmentation/01_prepare_dataset.py` |
| 3. Train supervised baseline | Cluster / Colab | `EXPERIMENT = 'supervised'` in notebook |
| 4. Pretrain MAE (self-supervised) | Cluster | `sbatch slurm/pretrain_selfsup.sh` |
| 5. Fine-tune on labeled data | Cluster / Colab | `EXPERIMENT = 'selfsup_finetune'` |
| 6. Train semi-supervised | Cluster | `sbatch slurm/train_semisup.sh` |
| 7. Train few-shot model | Cluster / Colab | `EXPERIMENT = 'fewshot'` |
| 8. Compare all methods | Local / Colab | `scripts/06_compare_methods.py` (TODO) |
| 9. Generate LLM reports | Local | `python llm/generate_report.py` |

---

## Experiments

Each experiment has its own config in `configs/` and saves output to `experiments/<name>/`:

| Experiment | Config | Key hyperparameter |
|------------|--------|--------------------|
| Supervised baseline | `supervised_baseline.yaml` | 100% labeled data |
| Semi-supervised | `semisup.yaml` | `labeled_ratio: 0.10` (10%) |
| Self-supervised pretrain | `selfsup_pretrain.yaml` | `mask_ratio: 0.75` |
| Few-shot | `fewshot.yaml` | `n_shot: 5` |

To resume training after a Colab session drop, set `RESUME_FROM` in the notebook to the last saved checkpoint path.

---

## Monitoring

TensorBoard logs are saved to `experiments/<name>/logs/`. In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/tesi/experiments
```

---

## Data

The project uses the **Medical Segmentation Decathlon Task07_Pancreas** dataset:
- 281 abdominal CT scans with pancreas + tumor annotations
- Labels: 0 = background, 1 = pancreas parenchyma, 2 = tumor
- Download: [medicaldecathlon.com](http://medicaldecathlon.com)

Place the downloaded folder at `data/Task07_Pancreas/`, then run `segmentation/01_prepare_dataset.py` to convert it to the required format.
