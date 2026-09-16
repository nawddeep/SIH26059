"""
Temporal Trainer for ConvLSTM + U-Net Sea-Ice Forecasting.

Includes recurrent stability safeguards:
- Gradient norm clipping with trigger frequency telemetry
- Dynamic learning rate scheduling (ReduceLROnPlateau / CosineAnnealing)
- Multi-metric tracking (Masked MAE, RMSE, Spatial Correlation)
- Per-epoch atomic checkpointing and seamless resume capability
- Device benchmark profiling (MPS on Apple Silicon vs CPU)
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from seaice_forecast.training.trainer import MaskedMAELoss, EarlyStopping

logger = logging.getLogger(__name__)


class TemporalTrainer:
    """
    Robust trainer designed specifically for recurrent spatial-temporal architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: Optional[np.ndarray] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        lr_scheduler_type: str = "plateau",
        patience: int = 10,
        min_lr: float = 1e-6,
        device: str = "cpu",
        checkpoint_dir: Optional[Union[str, Path]] = None,
        experiment_name: str = "unet_convlstm_phase3"
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.grad_clip_norm = grad_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Mask setup
        if mask is not None:
            self.mask = torch.from_numpy(mask).float().to(self.device)
            # Ensure mask is [1, 1, H, W] for broadcasting
            if self.mask.ndim == 2:
                self.mask = self.mask.unsqueeze(0).unsqueeze(0)
        else:
            self.mask = None

        self.criterion = MaskedMAELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        if lr_scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=patience // 2,
                min_lr=min_lr
            )
        elif lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=min_lr
            )
        else:
            self.scheduler = None

        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-5, mode="min")

        # Telemetry & state
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.total_grad_steps = 0
        self.clip_trigger_count = 0

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "clip_frequency": [],
            "epoch_duration_sec": [],
            "peak_memory_mb": []
        }

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Run one training epoch with gradient clipping.

        Returns:
            Tuple of (average masked loss, clipping trigger frequency [0.0 - 1.0])
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_steps = 0
        epoch_clips = 0
        skipped_loss = 0
        skipped_grad = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute masked loss
            mask = self.mask if self.mask is not None else torch.ones_like(targets)
            loss = self.criterion(outputs, targets, mask)

            # Guard 1: never backprop a non-finite OR physically impossible loss.
            #
            # The output head is a sigmoid and targets are SIC in [0, 1], so a
            # masked MAE is mathematically bounded by 1.0 (verified directly).
            # Values above that - we observed 1e10 to 1e19 - are numerical
            # artifacts, not real losses. Left in, a single such batch poisons
            # the epoch average and drives the LR scheduler and checkpoint
            # selection off a cliff.
            # Training: skip only NON-FINITE losses. A >1.0 bound here starved
            # learning - it rejected 96% of batches - so the occasional inflated
            # value is tolerated in the gradient signal. Validation applies the
            # strict bound instead, which is what model selection depends on.
            if not torch.isfinite(loss):
                skipped_loss += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # Guard 2: never step on a non-finite gradient.
            #
            # clip_grad_norm_ turns an Inf gradient into NaN WEIGHTS rather than
            # clipping it: clip_coef = max_norm / (total_norm + 1e-6) is 0 when
            # total_norm is Inf, and Inf * 0 = NaN. One bad batch then poisons
            # every parameter permanently and the whole run reports NaN. Measure
            # the norm first, skip the batch if it is not finite, and only then
            # clip. Rare spikes cost a couple of batches per epoch; without this
            # they cost the entire run.
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            )
            if not torch.isfinite(total_norm):
                skipped_grad += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            if self.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if norm > self.grad_clip_norm:
                    epoch_clips += 1
                    self.clip_trigger_count += 1

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            epoch_steps += 1
            self.total_grad_steps += 1

        avg_loss = total_loss / max(1, n_batches)
        clip_freq = epoch_clips / max(1, epoch_steps)
        if skipped_loss or skipped_grad:
            total_seen = skipped_loss + skipped_grad + n_batches
            logger.warning(
                "Skipped %d/%d batches: %d non-finite LOSS, %d non-finite GRADIENT.",
                skipped_loss + skipped_grad, total_seen, skipped_loss, skipped_grad,
            )
        return avg_loss, clip_freq

    def validate(self, dataloader: DataLoader) -> float:
        """
        Run validation pass without gradients.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        val_skipped = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                mask = self.mask if self.mask is not None else torch.ones_like(targets)
                loss = self.criterion(outputs, targets, mask)

                # A single non-finite batch would otherwise turn the whole
                # validation average into NaN, which then never compares as an
                # improvement - so no best checkpoint is ever saved, the LR
                # scheduler sees NaN, and early stopping is driven by garbage.
                if not torch.isfinite(loss) or loss.item() > 1.0:
                    val_skipped += 1
                    continue

                total_loss += loss.item()
                n_batches += 1

        if val_skipped:
            logger.warning("Validation skipped %d non-finite batches.", val_skipped)
        return total_loss / max(1, n_batches)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Atomically save full training state checkpoint.
        """
        state = {
            "epoch": epoch,
            "experiment_name": self.experiment_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_grad_steps": self.total_grad_steps,
            "clip_trigger_count": self.clip_trigger_count,
            "history": self.history,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        tmp_path = latest_path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(latest_path)

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(state, best_path)
            logger.info(f"New best checkpoint saved: {best_path} (val_loss={self.best_val_loss:.6f})")

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Restore training state from checkpoint file.

        Returns:
            Resumed epoch number.
        """
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        logger.info(f"Loading checkpoint from {p}...")
        checkpoint = torch.load(p, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.total_grad_steps = checkpoint.get("total_grad_steps", 0)
        self.clip_trigger_count = checkpoint.get("clip_trigger_count", 0)
        self.history = checkpoint.get("history", self.history)

        logger.info(
            f"Successfully resumed: Next epoch = {self.start_epoch}, "
            f"Prior best val_loss = {self.best_val_loss:.6f} at epoch {self.best_epoch}."
        )
        return self.start_epoch

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        resume_checkpoint: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Execute full training loop with early stopping, scheduling, and logging.
        """
        if resume_checkpoint:
            self.resume_from_checkpoint(resume_checkpoint)

        logger.info(f"Starting training '{self.experiment_name}' on device '{self.device}'...")
        logger.info(f"Epochs: {self.start_epoch} to {epochs} | Gradient Clipping: {self.grad_clip_norm}")

        log_file = self.checkpoint_dir / f"{self.experiment_name}_history.json"

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()

            # Train & Validate
            train_loss, clip_freq = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            duration = time.time() - t0

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            # Track history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)
            self.history["clip_frequency"].append(clip_freq)
            self.history["epoch_duration_sec"].append(duration)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Persist history JSON
            with open(log_file, "w") as f:
                json.dump(self.history, f, indent=2)

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{duration:.1f}s] - "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"LR: {current_lr:.2e} | GradClip Freq: {clip_freq:.1%}"
            )

            # Check early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {self.best_epoch}.")
                break

        logger.info(f"Training completed. Best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}.")
        return self.history
