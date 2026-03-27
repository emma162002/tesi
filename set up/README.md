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
