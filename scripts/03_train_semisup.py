"""
Entry point for semi-supervised training.

Usage:
    python scripts/03_train_semisup.py --config configs/semisup.yaml --output experiments/semisup
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_config
from src.data.dataset import SemiSupervisedDataset
from src.training.trainer_semisup import SemiSupervisedTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--resume",  default=None)
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = cfg["data"]
    ss_cfg   = cfg["semisup"]

    ds = SemiSupervisedDataset(
        data_dir       = data_cfg["data_dir"],
        labeled_ratio  = ss_cfg["labeled_ratio"],
        val_fold       = data_cfg["val_fold"],
        num_folds      = data_cfg["num_folds"],
        unlabeled_dir  = data_cfg.get("unlabeled_dir"),
        patch_size     = tuple(data_cfg["patch_size"]),
        cache_rate     = data_cfg.get("cache_rate", 0.0),
    )
    bs  = cfg["training"]["batch_size"]
    nw  = cfg["training"]["num_workers"]

    trainer = SemiSupervisedTrainer(cfg, args.output, device=args.device, resume=args.resume)
    trainer.train(
        labeled_loader   = ds.get_labeled_loader(batch_size=bs, num_workers=nw),
        unlabeled_loader = ds.get_unlabeled_loader(batch_size=bs, num_workers=nw),
        val_loader       = ds.get_val_loader(num_workers=nw),
    )


if __name__ == "__main__":
    main()
