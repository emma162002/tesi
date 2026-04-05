"""
Entry point for self-supervised MAE pretraining on unlabeled CT data.

Usage:
    python scripts/02_pretrain_selfsup.py --config configs/selfsup_pretrain.yaml --output experiments/selfsup_pretrain
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from monai.data import Dataset as MonaiDataset

from src.utils.io import load_config
from src.data.transforms import get_unlabeled_transforms
from src.training.trainer_selfsup import SelfSupervisedPretrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = cfg["data"]
    patch_size = tuple(data_cfg["patch_size"])

    unlabeled_dir = data_cfg.get("unlabeled_dir") or (data_cfg["data_dir"] + "/imagesTr")
    imgs  = sorted(Path(unlabeled_dir).glob("*.nii.gz"))
    data  = [{"image": str(p)} for p in imgs]
    print(f"Unlabeled CT volumes: {len(data)}")

    ds = MonaiDataset(data=data, transform=get_unlabeled_transforms(patch_size))
    loader = DataLoader(
        ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["training"]["num_workers"],
        pin_memory  = torch.cuda.is_available(),
    )

    trainer = SelfSupervisedPretrainer(cfg, args.output, device=args.device)
    trainer.train(loader)


if __name__ == "__main__":
    main()
