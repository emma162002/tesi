"""
Self-supervised pretraining trainer using 3D Masked Autoencoding (MAE).

Pretraining phase:
  - Feed unlabeled CT patches to the MAE
  - Randomly mask 75% of 3D patches
  - Train encoder + decoder to reconstruct original intensities
  - Save encoder weights for downstream fine-tuning

Fine-tuning phase:
  - Load pretrained encoder into PancreasUNet
  - Train normally with supervised loss on labeled data
  - Encoder LR is scaled down (lower LR preserves pretrained features)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from ..models.masked_autoencoder import MaskedAutoencoder3D
from ..models.unet import build_model
from ..losses.dice_ce import build_loss
from ..utils.io import save_checkpoint, load_checkpoint
from ..evaluation.metrics import evaluate_case, aggregate_metrics
from ..data.transforms import PATCH
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete


class SelfSupervisedPretrainer:
    """
    MAE pretraining on unlabeled CT data.

    Args:
        cfg:        config dict (selfsup_pretrain.yaml)
        output_dir: where to save the pretrained encoder
        device:     'cuda' | 'cpu'
    """

    def __init__(self, cfg: dict, output_dir: str, device: str = "cuda"):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        mae_cfg = cfg.get("mae", {})
        self.model = MaskedAutoencoder3D(
            in_channels  = mae_cfg.get("in_channels", 1),
            enc_features = tuple(mae_cfg.get("enc_features", [32, 64, 128, 256, 512])),
            dec_features = tuple(mae_cfg.get("dec_features", [256, 128, 64, 32])),
            patch_size   = mae_cfg.get("patch_size", 16),
            mask_ratio   = mae_cfg.get("mask_ratio", 0.75),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[MAE Pretrainer] Parameters: {n_params:,}")

        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = opt_cfg.get("lr", 1e-3),
            weight_decay = opt_cfg.get("weight_decay", 0.05),
            betas        = (0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max   = cfg["training"].get("epochs", 200),
            eta_min = 1e-6,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

    def train(self, unlabeled_loader, epochs: Optional[int] = None):
        n_epochs = epochs or self.cfg["training"].get("epochs", 200)
        print(f"\n[MAE Pretrainer] Pretraining for {n_epochs} epochs on unlabeled CT\n")

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            n_steps    = 0

            for batch in unlabeled_loader:
                imgs = batch["image"].to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    loss_dict = self.model.compute_loss(imgs)
                    loss      = loss_dict["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_steps    += 1

            self.scheduler.step()
            avg_loss = total_loss / max(n_steps, 1)
            self.writer.add_scalar("pretrain/recon_loss", avg_loss, epoch)

            if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
                lr = self.scheduler.get_last_lr()[0]
                print(f"Pretrain Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Recon Loss: {avg_loss:.5f} | LR: {lr:.2e}")

                # Save encoder checkpoint
                encoder_path = str(self.output_dir / f"encoder_epoch{epoch+1}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "encoder_state":    self.model.encoder.state_dict(),
                    "recon_loss":       avg_loss,
                }, encoder_path)

        # Save final encoder
        final_path = str(self.output_dir / "encoder_pretrained.pth")
        torch.save({
            "epoch":          n_epochs,
            "encoder_state":  self.model.encoder.state_dict(),
            "mae_state":      self.model.state_dict(),
            "recon_loss":     avg_loss,
        }, final_path)
        print(f"\n[MAE Pretrainer] Pretrained encoder saved → {final_path}")
        self.writer.close()
        return final_path


class SelfSupFineTuner:
    """
    Fine-tuning trainer that loads a pretrained MAE encoder
    and fine-tunes the full U-Net with labeled data.

    The encoder LR is divided by lr_multiplier to preserve pretrained features.

    Args:
        cfg:             config dict (selfsup_pretrain.yaml)
        encoder_ckpt:    path to pretrained encoder checkpoint
        output_dir:      output directory
        device:          'cuda' | 'cpu'
        lr_multiplier:   encoder LR = base LR / lr_multiplier (default 10x smaller)
    """

    def __init__(
        self,
        cfg:           dict,
        encoder_ckpt:  str,
        output_dir:    str,
        device:        str   = "cuda",
        lr_multiplier: float = 10.0,
    ):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build model with pretrained encoder
        model_cfg = cfg["model"].copy()
        model_cfg["pretrained_encoder"] = encoder_ckpt
        self.model = build_model(model_cfg).to(self.device)

        self.criterion = build_loss(cfg["loss"])

        # Differential LR: encoder gets lower LR
        opt_cfg  = cfg.get("optimizer", {})
        base_lr  = opt_cfg.get("lr", 1e-4)
        enc_lr   = base_lr / lr_multiplier

        # Separate encoder and decoder parameters
        # (MONAI UNet: encoder is model.net.model[0..n_down-1])
        all_params    = list(self.model.named_parameters())
        enc_params    = [p for n, p in all_params if "model.0" in n or "model.1" in n]
        dec_params    = [p for n, p in all_params if "model.0" not in n and "model.1" not in n]

        self.optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": enc_lr},
            {"params": dec_params, "lr": base_lr},
        ], weight_decay=opt_cfg.get("weight_decay", 1e-5))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max   = cfg["training"].get("epochs", 300),
            eta_min = 1e-6,
        )
        self.scaler     = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=cfg["model"].get("num_classes", 3))
        self.post_label = AsDiscrete(to_onehot=cfg["model"].get("num_classes", 3))
        self.writer     = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        self.best_dice  = -1.0
        self.best_ckpt  = str(self.output_dir / "best_model.pth")

        print(f"[FineTuner] Encoder LR: {enc_lr:.2e} | Decoder LR: {base_lr:.2e}")

    def train(self, train_loader, val_loader, epochs: Optional[int] = None):
        from .trainer_supervised import SupervisedTrainer
        # Reuse supervised training loop logic
        n_epochs = epochs or self.cfg["training"].get("epochs", 300)
        val_freq = self.cfg["training"].get("val_every", 5)

        print(f"\n[FineTuner] Fine-tuning for {n_epochs} epochs\n")

        for epoch in range(n_epochs):
            t0 = time.time()
            self.model.train()
            total_loss = 0.0
            n_steps    = 0

            for batch in train_loader:
                imgs   = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    logits    = self.model(imgs)
                    loss_dict = self.criterion(logits, labels)
                    loss      = loss_dict["loss"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_steps    += 1

            self.scheduler.step()
            self.writer.add_scalar("train/loss", total_loss / max(n_steps, 1), epoch)

            if (epoch + 1) % val_freq == 0 or epoch == n_epochs - 1:
                val_metrics = self._val_epoch(val_loader, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                tumor_dice = val_metrics.get("tumor_dice_mean", 0.0)
                print(
                    f"Epoch {epoch+1:3d}/{n_epochs} | "
                    f"Loss: {total_loss/max(n_steps,1):.4f} | "
                    f"Tumor Dice: {tumor_dice:.4f} | "
                    f"Time: {time.time()-t0:.1f}s"
                )
                if tumor_dice > self.best_dice:
                    self.best_dice = tumor_dice
                    save_checkpoint(self.model, self.optimizer, epoch,
                                    val_metrics, self.best_ckpt, self.scheduler)
                    print(f"  ★ New best: {tumor_dice:.4f}")

        self.writer.close()
        return self.best_ckpt

    @torch.no_grad()
    def _val_epoch(self, val_loader, epoch: int) -> dict:
        self.model.eval()
        patch_size = tuple(self.cfg["model"].get("img_size", list(PATCH)))
        per_case   = []

        for batch in val_loader:
            imgs   = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = sliding_window_inference(imgs, patch_size, 2, self.model, overlap=0.5)
            preds  = [self.post_pred(p).argmax(0).cpu().numpy()
                      for p in decollate_batch(logits)]
            gts    = [self.post_label(l).argmax(0).cpu().numpy()
                      for l in decollate_batch(labels)]
            for pred, gt in zip(preds, gts):
                per_case.append(evaluate_case(pred, gt,
                    classes={1: "pancreas", 2: "tumor"},
                    compute_surface_metrics=(epoch % 10 == 0)))

        return aggregate_metrics(per_case)
