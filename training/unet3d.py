import math
import os
import time
from typing import Any, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

try:
    from .losses import soft_dice_cldice, DiceBCELoss, DiceFocalLoss, BCEClDiceLoss
except ImportError:
    from losses import soft_dice_cldice, DiceBCELoss, DiceFocalLoss, BCEClDiceLoss


class DoubleConv3D(nn.Module):
    """Two 3x3x3 conv + InstanceNorm + LeakyReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class FlexibleUNet3D(pl.LightningModule):
    """
    3D UNet for volumetric segmentation.

    Input:  (B, 1, D, H, W)  -- single-channel grayscale volume
    Output: (B, 1, D, H, W)  -- binary segmentation logits
    """

    def __init__(
        self,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 1,
        min_features: int = 32,
        max_features: int = 320,
        num_stages: int = 5,
        loss: str = "BCE",
        freeze_encoder: bool = False,
        pos_weight: float = 1.0,
        in_channels: int = 1,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = loss
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.freeze_encoder = freeze_encoder
        self.pos_weight = pos_weight
        self.in_channels = in_channels
        self._last_train_step_compute_s = 0.0
        self._log_3d_val_images = os.environ.get("SYNTRACT_LOG_3D_VAL_IMAGES", "0") == "1"
        self._log_3d_val_every_n_epochs = max(1, int(os.environ.get("SYNTRACT_LOG_3D_VAL_EVERY", "10")))
        self.save_hyperparameters()

        # Build feature list for each stage
        # For 3D we use fewer stages and cap max_features lower to manage memory
        self.features = [
            min(max_features, min_features * (2 ** i))
            for i in range(num_stages)
        ]

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        ch = in_channels
        for feat in self.features:
            self.encoder_blocks.append(DoubleConv3D(ch, feat))
            ch = feat

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in range(len(self.features) - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(
                    self.features[i], self.features[i - 1],
                    kernel_size=2, stride=2,
                )
            )
            self.decoder_blocks.append(
                DoubleConv3D(self.features[i - 1] * 2, self.features[i - 1])
            )

        # Final 1x1x1 conv
        self.final_conv = nn.Conv3d(self.features[0], 1, kernel_size=1)

        # Loss
        if self.loss == "BCE":
            self.criterion = DiceBCELoss(pos_weight=self.pos_weight)
        elif self.loss == "focal":
            self.criterion = DiceFocalLoss(pos_weight=self.pos_weight)
        elif self.loss == "cldice":
            self.criterion = soft_dice_cldice()
        elif self.loss == "bce_cldice":
            # DiceBCE (keeps pos_weight) + clDice topology term. Takes logits, so
            # it uses the standard (non-sigmoid) loss branch in _shared_step.
            self.criterion = BCEClDiceLoss(pos_weight=self.pos_weight)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

        # Removed stateful metric objects - compute IoU/Dice directly per-step
        # This fixes the train_iou < val_iou anomaly caused by metric state accumulation issues

        if self.freeze_encoder:
            self._freeze_encoder()

    # ------------------------------------------------------------------
    def _freeze_encoder(self) -> None:
        print("Freezing encoder layers for finetuning...")
        for param in self.encoder_blocks.parameters():
            param.requires_grad = False
        print(
            f"Encoder frozen. Trainable parameters: "
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) volume -> (B, 1, D, H, W) segmentation logits."""
        if x.ndim != 5:
            raise ValueError(
                f"FlexibleUNet3D expects a 5-D (B, C, D, H, W) tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"FlexibleUNet3D was built for {self.in_channels} input channel(s), "
                f"got {x.shape[1]} (input shape {tuple(x.shape)})"
            )
        d, h, w = x.shape[2:]

        # Determine how many pooling ops we can do
        num_pools = 0
        cd, ch_, cw = d, h, w
        for _ in range(len(self.features) - 1):
            if cd >= 2 and ch_ >= 2 and cw >= 2:
                cd, ch_, cw = cd // 2, ch_ // 2, cw // 2
                num_pools += 1
            else:
                break

        # Encoder
        encoder_outputs = []
        x = self.encoder_blocks[0](x)
        encoder_outputs.append(x)

        for i in range(1, num_pools + 1):
            x = F.max_pool3d(x, 2)
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)

        # Decoder
        skip_features = encoder_outputs[:-1][::-1]
        for i in range(len(skip_features)):
            x = self.upconvs[i](x)
            # Handle dimension mismatches from rounding
            if x.shape[2:] != skip_features[i].shape[2:]:
                x = F.interpolate(
                    x, skip_features[i].shape[2:],
                    mode="trilinear", align_corners=True,
                )
            x = torch.cat([x, skip_features[i]], dim=1)
            x = self.decoder_blocks[i](x)

        return self.final_conv(x)

    # ------------------------------------------------------------------
    def _infer_total_training_steps(self) -> int:
        trainer = self.trainer
        if trainer is None:
            return -1

        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        if max_epochs <= 0:
            return -1

        dm = getattr(trainer, "datamodule", None)
        batches_per_epoch = None

        if dm is not None and bool(getattr(dm, "on_the_fly", False)):
            dm_batches = getattr(dm, "train_batches_per_epoch", None)
            if isinstance(dm_batches, int) and dm_batches > 0:
                batches_per_epoch = dm_batches

        if batches_per_epoch is None:
            train_dataset = getattr(dm, "train_dataset", None) if dm is not None else None
            dataset_batches = getattr(train_dataset, "batches_per_epoch", None)
            if isinstance(dataset_batches, int) and dataset_batches > 0:
                batches_per_epoch = dataset_batches

        if batches_per_epoch is None:
            train_dataset = getattr(dm, "train_dataset", None) if dm is not None else None
            batch_size = int(getattr(dm, "batch_size", self.batch_size) or self.batch_size or 1)
            try:
                dataset_len = len(train_dataset) if train_dataset is not None else 0
            except TypeError:
                dataset_len = 0
            if dataset_len > 0:
                batches_per_epoch = math.ceil(dataset_len / max(1, batch_size))

        if batches_per_epoch is None:
            trainer_batches = getattr(trainer, "num_training_batches", None)
            if isinstance(trainer_batches, int) and trainer_batches > 0:
                batches_per_epoch = trainer_batches

        if batches_per_epoch is None:
            return -1

        accumulate = getattr(trainer, "accumulate_grad_batches", 1)
        if not isinstance(accumulate, int):
            accumulate = 1
        steps_per_epoch = math.ceil(int(batches_per_epoch) / max(1, accumulate))
        return int(steps_per_epoch * max_epochs)

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = self._infer_total_training_steps()
        if total_steps <= 0:
            print("Scheduler disabled: could not infer a positive total_steps for cosine warmup.")
            return optimizer

        max_epochs = max(1, int(getattr(self.trainer, "max_epochs", 1) or 1))
        steps_per_epoch = max(1, math.ceil(total_steps / max_epochs))
        warmup_steps = max(0, int(steps_per_epoch * self.warmup_epochs))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"Scheduler configured: total_steps={total_steps}, warmup_steps={warmup_steps}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward + loss + stateless metrics. Returns (loss, binary preds, target, input)."""
        volume, seg_mask = batch
        logits = self(volume)

        # Ensure mask has channel dim: (B, D, H, W) -> (B, 1, D, H, W)
        if seg_mask.ndim == 4:
            seg_mask = seg_mask.unsqueeze(1)
        seg_mask = seg_mask.float()
        if seg_mask.shape != logits.shape:
            raise ValueError(
                f"Segmentation mask shape {tuple(seg_mask.shape)} does not match model output "
                f"{tuple(logits.shape)} (input {tuple(volume.shape)})"
            )

        if self.loss == "cldice":
            probs = torch.sigmoid(logits)
            loss = self.criterion(seg_mask, probs)  # soft_dice_cldice(y_true, y_pred)
        else:
            loss = self.criterion(logits, seg_mask)
            probs = torch.sigmoid(logits)
        pred_mask = probs > 0.5

        # Compute IoU and Dice manually (stateless) to avoid metric state issues
        target_int = seg_mask.int()
        tp = ((pred_mask.int() == 1) & (target_int == 1)).sum().float()
        fp = ((pred_mask.int() == 1) & (target_int == 0)).sum().float()
        fn = ((pred_mask.int() == 0) & (target_int == 1)).sum().float()

        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-6)
        # Dice = 2*TP / (2*TP + FP + FN)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        # Soft (threshold-free) dice -- tracks model quality even when the
        # current 0.5 threshold is miscalibrated for the current pos_weight.
        probs_f = probs.float()
        soft_inter = (probs_f * seg_mask).sum()
        soft_dice = (2.0 * soft_inter) / (probs_f.sum() + seg_mask.sum() + 1e-6)

        sync_dist = bool(self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=sync_dist)
        self.log(f"{stage}_iou", iou, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
        self.log(f"{stage}_dice", dice, on_epoch=True, prog_bar=True, logger=True, sync_dist=sync_dist)
        self.log(f"{stage}_soft_dice", soft_dice, on_epoch=True, prog_bar=False, logger=True, sync_dist=sync_dist)
        self.log(f"{stage}_target_pos_frac", seg_mask.mean(), on_epoch=True, logger=True, sync_dist=sync_dist)
        self.log(f"{stage}_pred_pos_frac", pred_mask.float().mean(), on_epoch=True, logger=True, sync_dist=sync_dist)
        self.log(f"{stage}_prob_mean", probs_f.mean(), on_epoch=True, logger=True, sync_dist=sync_dist)
        return loss, pred_mask, seg_mask, volume

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        step_start = time.perf_counter()
        loss, *_ = self._shared_step(batch, "train")
        self._last_train_step_compute_s = time.perf_counter() - step_start
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, pred_mask, seg_mask, volume = self._shared_step(batch, "val")

        # Log 3D volume visualization for WandB
        should_log_images = (
            self._log_3d_val_images
            and batch_idx == 0
            and self.current_epoch % self._log_3d_val_every_n_epochs == 0
            and self.logger is not None
            and hasattr(self.logger, "experiment")
        )
        if should_log_images:
            try:
                import wandb as _wandb

                # Extract 3D volumes (B, C, D, H, W) -> (D, H, W)
                input_vol = volume[0, 0].cpu().numpy()  # Full 3D volume
                pred_vol = (pred_mask[0, 0].cpu().numpy() * 255).astype("uint8")
                gt_vol = (seg_mask[0, 0].cpu().numpy() * 255).astype("uint8")

                # Create multi-slice visualization: 3 orthogonal planes
                d, h, w = input_vol.shape
                mid_d, mid_h, mid_w = d // 2, h // 2, w // 2

                # Normalize input for visualization
                input_norm = input_vol.copy()
                input_norm = (input_norm - input_norm.min()) / (input_norm.max() - input_norm.min() + 1e-8)
                input_norm = (input_norm * 255).astype("uint8")

                # Axial slice (z-plane, middle depth)
                axial_input = input_norm[mid_d]
                axial_pred = pred_vol[mid_d]
                axial_gt = gt_vol[mid_d]

                # Coronal slice (y-plane, middle height)
                coronal_input = input_norm[:, mid_h, :]
                coronal_pred = pred_vol[:, mid_h, :]
                coronal_gt = gt_vol[:, mid_h, :]

                # Sagittal slice (x-plane, middle width)
                sagittal_input = input_norm[:, :, mid_w]
                sagittal_pred = pred_vol[:, :, mid_w]
                sagittal_gt = gt_vol[:, :, mid_w]

                self.logger.experiment.log({
                    "3d_axial_input": _wandb.Image(axial_input, caption="Axial (depth)"),
                    "3d_axial_pred": _wandb.Image(axial_pred, caption="Axial Pred (depth)"),
                    "3d_axial_gt": _wandb.Image(axial_gt, caption="Axial GT (depth)"),
                    "3d_coronal_input": _wandb.Image(coronal_input, caption="Coronal (height)"),
                    "3d_coronal_pred": _wandb.Image(coronal_pred, caption="Coronal Pred (height)"),
                    "3d_coronal_gt": _wandb.Image(coronal_gt, caption="Coronal GT (height)"),
                    "3d_sagittal_input": _wandb.Image(sagittal_input, caption="Sagittal (width)"),
                    "3d_sagittal_pred": _wandb.Image(sagittal_pred, caption="Sagittal Pred (width)"),
                    "3d_sagittal_gt": _wandb.Image(sagittal_gt, caption="Sagittal GT (width)"),
                })
            except Exception as exc:
                print(f"Skipping 3D val image logging (wandb unavailable or failed): {exc}")
        return loss
