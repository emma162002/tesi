"""
Fully supervised baseline trainer.

This is the reference model — trained with full pixel-level annotations.
All other methods (semi-supervised, self-supervised, few-shot) are compared against it.

Supports:
  - Single GPU (Colab T4/A100)
  - Automatic mixed precision (AMP) for memory efficiency
  - Checkpoint saving / resuming (important for Colab session drops)
  - TensorBoard logging
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

from ..models.unet import build_model
from ..losses.dice_ce import build_loss, DiceCECombined
from ..evaluation.metrics import evaluate_case, aggregate_metrics
from ..utils.io import save_checkpoint, load_checkpoint, load_config, save_results_json
from ..data.transforms import PATCH


class SupervisedTrainer:
    """
    Supervised segmentation trainer.

    Args:
        cfg:         config dict (loaded from YAML)
        output_dir:  where to save checkpoints and logs
        device:      'cuda' | 'cpu' | 'cuda:0'
        resume:      path to checkpoint to resume from
    """

    def __init__(
        self,
        cfg:        dict,
        output_dir: str,
        device:     str  = "cuda",
        resume:     Optional[str] = None,
    ):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Trainer] Device: {self.device}")
        if self.device.type == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")

        # Model
        self.model = build_model(cfg["model"]).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Model parameters: {n_params:,}")

        # Loss
        self.criterion = build_loss(cfg["loss"])

        # Optimizer
        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr            = opt_cfg.get("lr", 1e-4),
            weight_decay  = opt_cfg.get("weight_decay", 1e-5),
        )

        # LR scheduler: cosine annealing with warm restarts
        sched_cfg = cfg.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max  = cfg["training"].get("epochs", 300),
            eta_min = sched_cfg.get("eta_min", 1e-6),
        )

        # AMP scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        # Post-processing: convert logits to discrete labels
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=cfg["model"].get("num_classes", 3))
        self.post_label = AsDiscrete(to_onehot=cfg["model"].get("num_classes", 3))

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        # State
        self.start_epoch  = 0
        self.best_dice    = -1.0
        self.best_ckpt    = str(self.output_dir / "best_model.pth")

        if resume is not None:
            self.start_epoch, metrics = load_checkpoint(
                resume, self.model, self.optimizer, self.scheduler,
                device=str(self.device)
            )
            self.best_dice = metrics.get("tumor_dice_mean", -1.0)

    def _train_epoch(self, train_loader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            imgs   = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(imgs)
                loss_dict = self.criterion(logits, labels)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            # Gradient clipping — important for stable 3D segmentation training
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        self.scheduler.step()
        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, val_loader, epoch: int) -> dict:
        self.model.eval()
        patch_size = tuple(self.cfg["model"].get("img_size", list(PATCH)))
        num_classes = self.cfg["model"].get("num_classes", 3)
        per_case = []

        for batch in val_loader:
            imgs   = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Sliding window inference — handles arbitrary volume sizes
            logits = sliding_window_inference(
                inputs     = imgs,
                roi_size   = patch_size,
                sw_batch_size = 2,
                predictor  = self.model,
                overlap    = 0.5,
            )

            # Convert to class labels
            pred_list  = decollate_batch(logits)
            label_list = decollate_batch(labels)
            preds      = [self.post_pred(p).argmax(0).cpu().numpy()  for p in pred_list]
            gts        = [self.post_label(l).argmax(0).cpu().numpy() for l in label_list]

            spacing = batch.get("image_meta_dict", {}).get(
                "pixdim", [[1, 1, 1, 1]]
            )[0][1:4] if "image_meta_dict" in batch else (1.0, 1.0, 1.0)

            for pred, gt in zip(preds, gts):
                result = evaluate_case(
                    pred, gt,
                    spacing=tuple(float(s) for s in spacing),
                    classes={1: "pancreas", 2: "tumor"},
                    compute_surface_metrics=(epoch % 10 == 0),  # HD95 every 10 epochs
                )
                per_case.append(result)

        return aggregate_metrics(per_case)

    def train(self, train_loader, val_loader, epochs: Optional[int] = None):
        """Main training loop."""
        n_epochs = epochs or self.cfg["training"].get("epochs", 300)
        val_freq = self.cfg["training"].get("val_every", 5)

        print(f"\n[Trainer] Starting training for {n_epochs} epochs")
        print(f"[Trainer] Validating every {val_freq} epochs\n")

        for epoch in range(self.start_epoch, n_epochs):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(train_loader, epoch)

            # Log
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            # Validate
            if (epoch + 1) % val_freq == 0 or epoch == n_epochs - 1:
                val_metrics = self._val_epoch(val_loader, epoch)

                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                tumor_dice = val_metrics.get("tumor_dice_mean", 0.0)
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Loss: {train_metrics['train_loss']:.4f} | "
                    f"Tumor Dice: {tumor_dice:.4f} | "
                    f"Pancreas Dice: {val_metrics.get('pancreas_dice_mean', 0.0):.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

                # Save best model
                if tumor_dice > self.best_dice:
                    self.best_dice = tumor_dice
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_metrics, self.best_ckpt,
                        scheduler=self.scheduler
                    )
                    print(f"  ★ New best tumor Dice: {tumor_dice:.4f}")

            # Periodic checkpoint every 50 epochs (safety for Colab)
            if (epoch + 1) % 50 == 0:
                periodic_path = str(self.output_dir / f"checkpoint_epoch{epoch+1}.pth")
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    {"tumor_dice_mean": self.best_dice}, periodic_path
                )

        self.writer.close()
        print(f"\n[Trainer] Training complete. Best tumor Dice: {self.best_dice:.4f}")
        print(f"[Trainer] Best model saved at: {self.best_ckpt}")
        return self.best_ckpt
