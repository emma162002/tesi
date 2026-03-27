# Project Setup Guide

## Overview

This project has two main components:
1. **Pancreas segmentation** using nnUNet (pretrained model, fine-tuned on pancreas CT scans)
2. **Report generation** using a local LLM (via Ollama) that describes the segmentation results

---

## First-time setup (do this once)

### 1. Create the conda environment
The project requires Python 3.10 and several libraries. Run:
```bash
conda env create -f "set up/environment.yml"
```

### 2. Set environment variables
nnUNet needs to know where to find data and save results.
Add these lines to your `~/.zshrc` (or run them manually each session):
```bash
export nnUNet_raw="/Users/emmabattaglia/tesi/data/nnunet_raw"
export nnUNet_preprocessed="/Users/emmabattaglia/tesi/data/nnunet_preprocessed"
export nnUNet_results="/Users/emmabattaglia/tesi/data/nnunet_results"
```
Then reload: `source ~/.zshrc`

### 3. Install Ollama
Ollama is used to run the LLM locally.
```bash
brew install ollama
```
Then download your chosen model (e.g.):
```bash
ollama pull biomistral   # medical model, ~4GB
ollama pull llama3.1     # general purpose
```

---

## Every time you work on the project

### Activate the environment
Always do this before running any script:
```bash
conda activate tesi
```
You will see `(tesi)` at the start of your terminal prompt.

### Start Ollama (if using a local LLM)
```bash
ollama serve &
```

### Keep code up to date (if working across two computers)
```bash
git pull   # download latest changes from GitHub
```

### Push changes to GitHub after working
```bash
git add .
git commit -m "description of what you changed"
git push
```

---

## Project structure

```
tesi/
├── set up/
│   ├── README.md            <- this file
│   ├── environment.yml      <- conda dependencies
│   └── setup_env.sh         <- environment variables to add to ~/.zshrc
│
├── data/                    <- ignored by git (too large)
│   ├── nnunet_raw/          <- place the downloaded dataset here
│   ├── nnunet_preprocessed/ <- filled automatically by nnUNet
│   └── nnunet_results/      <- filled automatically by nnUNet (trained models)
│
├── segmentation/
│   ├── 01_prepare_dataset.py  <- converts Task07_Pancreas to nnUNet format
│   ├── 02_train.py            <- runs nnUNet preprocessing / training / inference
│   └── 03_evaluate.py         <- computes Dice score and volume error
│
├── llm/
│   └── generate_report.py     <- generates radiology report from segmentation metrics
│
└── pipeline/
    └── run_pipeline.py        <- runs the full pipeline end-to-end
```

---

## Workflow (step by step)

| Step | Where | Script / Command |
|------|--------|-----------------|
| 1. Download dataset | local | medicaldecathlon.com → Task07_Pancreas |
| 2. Convert dataset | local | `python segmentation/01_prepare_dataset.py` |
| 3. Preprocess | university GPU | `python segmentation/02_train.py preprocess` |
| 4. Train nnUNet | university GPU | `python segmentation/02_train.py train` |
| 5. Run inference | university GPU | `python segmentation/02_train.py inference <in> <out>` |
| 6. Evaluate | local or GPU | `python segmentation/03_evaluate.py <pred> <gt>` |
| 7. Generate reports | local | `python llm/generate_report.py` |
| 8. Full pipeline | local or GPU | `python pipeline/run_pipeline.py` |

---

## Future work & thesis ideas

### Segmentation

- **Compare fine-tuning strategies** — full fine-tuning vs. freezing the encoder vs. LoRA-style adapters. Which gives the best Dice with limited data?
- **Data augmentation** — nnUNet already does augmentation, but you can experiment with custom strategies (elastic deformations, intensity shifts) for pancreas-specific challenges
- **Uncertainty estimation** — use nnUNet's test-time augmentation or MC Dropout to produce confidence maps alongside the segmentation. This is clinically relevant and makes for a strong thesis contribution
- **Multi-class segmentation** — Task07 has two labels (pancreas + cancer). Evaluating both separately and showing the model handles tumor detection is a nice result
- **External validation** — if you can get access to a second dataset (e.g. NIH Pancreas-CT, 82 cases), testing on it shows generalization

### Report generation (LLM)

- **Prompt engineering** — try different prompt structures and compare output quality. Simple changes in wording can have a big effect
- **Model comparison** — benchmark at least two models (e.g. BioMistral 7B vs. Meditron 70B or LLaMA 3.1) on the same inputs and compare reports qualitatively
- **Structured vs. free-text output** — experiment with asking the LLM to return JSON (volume, findings, conclusion) vs. plain prose. Structured output is easier to evaluate automatically
- **Hallucination detection** — check if the LLM invents findings not supported by the segmentation metrics. This is an open problem and worth discussing in the thesis
- **Human evaluation** — ask a radiologist (or your supervisor) to rate a sample of generated reports on accuracy, fluency, and clinical usefulness. Even 20-30 cases is enough for a meaningful evaluation

### Pipeline & evaluation

- **End-to-end evaluation metric** — define a combined score that rewards both good segmentation (Dice) and good report quality (e.g. BLEU, BERTScore, or human rating)
- **Visualization** — add a script that overlays the segmentation mask on the CT slice and saves a figure. Useful for the thesis document and for sanity-checking results
- **DICOM support** — the current pipeline uses NIfTI. Supporting DICOM input makes it more realistic for clinical use

### What makes a strong thesis

- **Clear baseline** — always compare against the pretrained nnUNet without fine-tuning. Your contribution needs a reference point
- **Ablation study** — change one thing at a time (e.g. number of training cases, fine-tuning epochs) and show the effect on Dice score
- **Error analysis** — show cases where the model fails and explain why. This shows depth of understanding
- **Clinical context** — briefly explain why pancreas segmentation is hard (small size, low contrast, variable shape) and why automation matters
