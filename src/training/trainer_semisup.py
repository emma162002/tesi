"""
Semi-supervised trainer using Mean Teacher + Pseudo-label generation.

Training loop per epoch:
  1. For each labeled batch:
       student_pred → supervised DiceCE loss
  2. For each unlabeled batch:
       teacher(x) → pseudo-labels + confidence mask
       student(augment(x)) → consistency loss + pseudo-label loss
  3. EMA update: teacher ← EMA(student)

Key design choices:
  - Labeled and unlabeled loaders are iterated jointly (zip)
    with the unlabeled loader repeated if it is longer
  - Consistency weight is ramped up slowly to avoid early collapse
  - Pseudo-labels are only used when teacher confidence > threshold
"""

from __future__ import annotations
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
from ..losses.dice_ce import build_loss
from ..losses.consistency import (
    SemiSupervisedLoss,
    create_teacher,
    update_ema,
    PseudoLabelLoss,
)
from ..evaluation.metrics import evaluate_case, aggregate_metrics
from ..utils.io import save_checkpoint, load_checkpoint, save_results_json
from ..data.transforms import PATCH


class SemiSupervisedTrainer:
    """
    Mean Teacher semi-supervised trainer for pancreatic tumor segmentation.

    Args:
        cfg:        config dict
        output_dir: output directory for checkpoints and logs
        device:     'cuda' | 'cpu'
        resume:     path to checkpoint to resume from
    """

    def __init__(
        self,
        cfg:        dict,
        output_dir: str,
        device:     str = "cuda",
        resume:     Optional[str] = None,
    ):
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        semisup_cfg = cfg.get("semisup", {})

        # Student model
        self.student = build_model(cfg["model"]).to(self.device)

        # Teacher model: EMA copy, not trained by optimizer
        self.teacher = create_teacher(self.student)
        self.ema_alpha = semisup_cfg.get("ema_alpha", 0.999)

        # Pseudo-label generator
        self.pseudo_gen = PseudoLabelLoss(
            confidence_threshold = semisup_cfg.get("confidence_threshold", 0.90),
        )

        # Combined semi-supervised loss
        supervised_loss = build_loss(cfg["loss"])
        self.criterion = SemiSupervisedLoss(
            supervised_loss      = supervised_loss,
            use_pseudo_labels    = semisup_cfg.get("use_pseudo_labels", True),
            consistency_weight   = semisup_cfg.get("consistency_weight", 1.0),
            pseudo_weight        = semisup_cfg.get("pseudo_weight", 0.5),
            confidence_threshold = semisup_cfg.get("confidence_threshold", 0.90),
            ramp_up_epochs       = semisup_cfg.get("ramp_up_epochs", 40),
        )

        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr           = opt_cfg.get("lr", 1e-4),
            weight_decay = opt_cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max   = cfg["training"].get("epochs", 300),
            eta_min = 1e-6,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.post_pred  = AsDiscrete(argmax=True, to_onehot=cfg["model"].get("num_classes", 3))
        self.post_label = AsDiscrete(to_onehot=cfg["model"].get("num_classes", 3))
        self.writer     = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.start_epoch = 0
        self.best_dice   = -1.0
        self.best_ckpt   = str(self.output_dir / "best_model.pth")

        if resume is not None:
            self.start_epoch, metrics = load_checkpoint(
                resume, self.student, self.optimizer, self.scheduler,
                device=str(self.device)
            )
            # sync teacher to resumed student
            self.teacher = create_teacher(self.student)
            self.best_dice = metrics.get("tumor_dice_mean", -1.0)

    def _train_epoch(self, labeled_loader, unlabeled_loader, epoch: int) -> dict:
        self.student.train()
        self.teacher.eval()

        metrics_accum = {
            "loss": 0.0, "sup_loss": 0.0,
            "consistency_loss": 0.0, "pseudo_loss": 0.0,
            "annotated_frac": 0.0,
        }
        n_steps = 0

        # Cycle through unlabeled data to match labeled length
        unlabeled_iter = iter(unlabeled_loader)

        for labeled_batch in labeled_loader:
            # Get labeled data
            imgs_l   = labeled_batch["image"].to(self.device)
            labels   = labeled_batch["label"].to(self.device)

            # Get unlabeled data (cycle if exhausted)
            try:
                unlab_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlab_batch = next(unlabeled_iter)
            imgs_u = unlab_batch["image"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                # Student forward on labeled
                student_logits_l = self.student(imgs_l)

                # Teacher forward on unlabeled (no grad)
                with torch.no_grad():
                    teacher_logits_u = self.teacher(imgs_u)

                # Student forward on unlabeled
                student_logits_u = self.student(imgs_u)

                # Generate pseudo-labels from teacher
                pseudo_labels, confident_mask = self.pseudo_gen.generate_pseudo_labels(
                    self.teacher, imgs_u
                )

                # Combined loss
                loss_dict = self.criterion(
                    student_labeled_logits   = student_logits_l,
                    labels                   = labels,
                    student_unlabeled_logits = student_logits_u,
                    teacher_unlabeled_logits = teacher_logits_u,
                    pseudo_labels            = pseudo_labels,
                    confident_mask           = confident_mask,
                    epoch                    = epoch,
                )

            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA teacher update
            update_ema(self.student, self.teacher, alpha=self.ema_alpha)

            for k in metrics_accum:
                val = loss_dict.get(k, 0.0)
                metrics_accum[k] += val.item() if torch.is_tensor(val) else val
            n_steps += 1

        self.scheduler.step()
        return {f"train_{k}": v / max(n_steps, 1) for k, v in metrics_accum.items()}

    @torch.no_grad()
    def _val_epoch(self, val_loader, epoch: int) -> dict:
        """Validate using the teacher model (more stable than student)."""
        self.teacher.eval()
        patch_size  = tuple(self.cfg["model"].get("img_size", list(PATCH)))
        num_classes = self.cfg["model"].get("num_classes", 3)
        per_case    = []

        for batch in val_loader:
            imgs   = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = sliding_window_inference(
                inputs        = imgs,
                roi_size      = patch_size,
                sw_batch_size = 2,
                predictor     = self.teacher,
                overlap       = 0.5,
            )
            pred_list  = decollate_batch(logits)
            label_list = decollate_batch(labels)
            preds = [self.post_pred(p).argmax(0).cpu().numpy()  for p in pred_list]
            gts   = [self.post_label(l).argmax(0).cpu().numpy() for l in label_list]

            for pred, gt in zip(preds, gts):
                per_case.append(evaluate_case(
                    pred, gt,
                    classes={1: "pancreas", 2: "tumor"},
                    compute_surface_metrics=(epoch % 10 == 0),
                ))

        return aggregate_metrics(per_case)

    def train(self, labeled_loader, unlabeled_loader, val_loader,
              epochs: Optional[int] = None):
        n_epochs = epochs or self.cfg["training"].get("epochs", 300)
        val_freq = self.cfg["training"].get("val_every", 5)

        labeled_ratio = self.cfg.get("semisup", {}).get("labeled_ratio", 0.1)
        print(f"\n[SemiSup Trainer] Labeled ratio: {labeled_ratio:.0%}")
        print(f"[SemiSup Trainer] EMA alpha: {self.ema_alpha}")
        print(f"[SemiSup Trainer] Training for {n_epochs} epochs\n")

        for epoch in range(self.start_epoch, n_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(labeled_loader, unlabeled_loader, epoch)

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
                    f"Sup: {train_metrics['train_sup_loss']:.4f} | "
                    f"Cons: {train_metrics['train_consistency_loss']:.4f} | "
                    f"Tumor Dice: {tumor_dice:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

                if tumor_dice > self.best_dice:
                    self.best_dice = tumor_dice
                    save_checkpoint(
                        self.student, self.optimizer, epoch,
                        val_metrics, self.best_ckpt, scheduler=self.scheduler
                    )
                    print(f"  ★ New best tumor Dice: {tumor_dice:.4f}")

            if (epoch + 1) % 50 == 0:
                save_checkpoint(
                    self.student, self.optimizer, epoch,
                    {"tumor_dice_mean": self.best_dice},
                    str(self.output_dir / f"checkpoint_epoch{epoch+1}.pth")
                )

        self.writer.close()
        print(f"\n[SemiSup] Done. Best tumor Dice: {self.best_dice:.4f}")
        return self.best_ckpt
