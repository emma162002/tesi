"""
Few-shot segmentation trainer using Prototypical Networks.

Training paradigm (episodic training):
  - Each training iteration = one episode
  - Episode: K support images + Q query images, all with labels
  - Model learns to segment query images using only the K support examples
  - At test time: given K=5 annotated examples, segment new cases

This simulates the clinical scenario where only a handful of annotated
examples are available for a rare tumor type or new imaging protocol.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..models.prototype_net import PrototypeSegmentationNet
from ..evaluation.metrics import evaluate_case, aggregate_metrics
from ..utils.io import save_checkpoint, load_checkpoint, save_results_json
from ..data.transforms import PATCH


class FewShotTrainer:
    """
    Episodic trainer for prototypical few-shot tumor segmentation.

    Args:
        cfg:          config dict (fewshot.yaml)
        output_dir:   output directory
        device:       'cuda' | 'cpu'
        encoder_ckpt: optional pretrained encoder to initialize the backbone
        resume:       path to checkpoint to resume from
    """

    def __init__(
        self,
        cfg:          dict,
        output_dir:   str,
        device:       str           = "cuda",
        encoder_ckpt: Optional[str] = None,
        resume:       Optional[str] = None,
    ):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fs_cfg = cfg.get("fewshot", {})
        self.n_shot     = fs_cfg.get("n_shot", 5)
        self.n_query    = fs_cfg.get("n_query", 4)

        self.model = PrototypeSegmentationNet(
            in_channels  = cfg["model"].get("in_channels", 1),
            num_classes  = cfg["model"].get("num_classes", 2),  # binary: bg + tumor
            encoder_ckpt = encoder_ckpt,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[FewShot Trainer] Parameters: {n_params:,}")
        print(f"[FewShot Trainer] {self.n_shot}-shot, {self.n_query}-query episodes")

        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = opt_cfg.get("lr", 1e-4),
            weight_decay = opt_cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = cfg["training"].get("epochs", 100) // 3,
            gamma     = 0.5,
        )
        self.scaler  = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.writer  = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.start_epoch = 0
        self.best_dice   = -1.0
        self.best_ckpt   = str(self.output_dir / "best_model.pth")

        if resume is not None:
            self.start_epoch, metrics = load_checkpoint(
                resume, self.model, self.optimizer, device=str(self.device)
            )
            self.best_dice = metrics.get("tumor_dice_mean", -1.0)

    def _train_epoch(self, train_loader, epoch: int) -> dict:
        self.model.train()
        total_loss  = 0.0
        total_dice  = 0.0
        n_episodes  = 0

        for episode in train_loader:
            # episode keys: support_images, support_labels, query_images, query_labels
            # shapes: (B, K, 1, H, W, D) — B=1 for episode training
            support_imgs   = episode["support_images"].squeeze(0).to(self.device)
            support_labels = episode["support_labels"].squeeze(0).to(self.device)
            query_imgs     = episode["query_images"].squeeze(0).to(self.device)
            query_labels   = episode["query_labels"].squeeze(0).to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                loss_dict = self.model.compute_loss(
                    support_imgs, support_labels,
                    query_imgs,   query_labels,
                )

            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss_dict["ce_loss"]
            total_dice += loss_dict["dice"]
            n_episodes += 1

        self.scheduler.step()
        return {
            "train_loss": total_loss / max(n_episodes, 1),
            "train_dice": total_dice / max(n_episodes, 1),
        }

    @torch.no_grad()
    def _val_epoch(self, val_loader, epoch: int) -> dict:
        """Validate on held-out episodes using the same episodic protocol."""
        self.model.eval()
        per_episode_dice = []

        for episode in val_loader:
            support_imgs   = episode["support_images"].squeeze(0).to(self.device)
            support_labels = episode["support_labels"].squeeze(0).to(self.device)
            query_imgs     = episode["query_images"].squeeze(0).to(self.device)
            query_labels   = episode["query_labels"].squeeze(0).to(self.device)

            logits   = self.model(support_imgs, support_labels, query_imgs)
            preds    = logits.argmax(dim=1).cpu().numpy()   # (Q, H, W, D)
            gt       = query_labels.squeeze(1).cpu().numpy()

            # Dice for tumor class (class 1 in binary setting)
            for pred, gt_single in zip(preds, gt):
                pred_bin = (pred == 1).astype(float)
                gt_bin   = (gt_single == 1).astype(float)
                inter    = (pred_bin * gt_bin).sum()
                dice     = float(2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-6))
                per_episode_dice.append(dice)

        import numpy as np
        return {
            "tumor_dice_mean": float(np.mean(per_episode_dice)),
            "tumor_dice_std":  float(np.std(per_episode_dice)),
        }

    def train(
        self,
        train_loader,
        val_loader,
        epochs: Optional[int] = None,
    ):
        n_epochs = epochs or self.cfg["training"].get("epochs", 100)
        val_freq = self.cfg["training"].get("val_every", 5)

        print(f"\n[FewShot Trainer] Episodic training for {n_epochs} epochs\n")

        for epoch in range(self.start_epoch, n_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, epoch)

            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            if (epoch + 1) % val_freq == 0 or epoch == n_epochs - 1:
                val_metrics = self._val_epoch(val_loader, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                tumor_dice = val_metrics.get("tumor_dice_mean", 0.0)
                elapsed    = time.time() - t0
                print(
                    f"Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Dice: {train_metrics['train_dice']:.4f} | "
                    f"Val Dice: {tumor_dice:.4f} ± {val_metrics.get('tumor_dice_std', 0):.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

                if tumor_dice > self.best_dice:
                    self.best_dice = tumor_dice
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        val_metrics, self.best_ckpt
                    )
                    print(f"  ★ New best: {tumor_dice:.4f}")

        self.writer.close()
        print(f"\n[FewShot] Done. Best tumor Dice: {self.best_dice:.4f}")
        return self.best_ckpt
