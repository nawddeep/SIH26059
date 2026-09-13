"""
Multi-Horizon Forecasting Module for Antarctic Sea-Ice Concentration.

Supports two distinct multi-horizon forecasting strategies:
1. Direct Multi-Head Forecasting:
   A spatial-temporal backbone produces rich representations, and dedicated projection heads
   directly predict each forecast horizon (e.g. +1, +3, +5, +7 days).
2. Recursive Autoregressive Rollout:
   Iteratively rolls out a single-step model (e.g. UNetConvLSTM) over multiple future steps,
   feeding the predicted SIC back into the input sequence at each autoregressive step.

Includes multi-horizon weighted loss, AdamW optimizer, cosine annealing, and checkpoint/resume.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime

from seaice_forecast.models.unet_convlstm import UNetConvLSTM
from seaice_forecast.training.trainer import MaskedMAELoss

logger = logging.getLogger(__name__)


class DirectMultiHeadUNetConvLSTM(nn.Module):
    """
    Direct Multi-Head Architecture.

    Uses a shared UNetConvLSTM backbone up to the penultimate decoder layer,
    then projects through dedicated 1x1 conv heads for each horizon h in [1, 3, 5, 7].
    """

    def __init__(
        self,
        in_channels: int = 7,
        horizons: List[int] = [1, 3, 5, 7],
        seq_len: int = 7,
        encoder_channels: List[int] = [32, 64, 128, 256],
        bottleneck_channels: Optional[int] = None,
        convlstm_layers: int = 1,
        dropout: float = 0.1,
        output_activation: str = "sigmoid"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.horizons = sorted(horizons)
        self.seq_len = seq_len

        # Shared backbone (encoders, bottleneck ConvLSTM, and decoders)
        self.backbone = UNetConvLSTM(
            in_channels=in_channels,
            output_channels=1,
            seq_len=seq_len,
            encoder_channels=encoder_channels,
            bottleneck_channels=bottleneck_channels,
            convlstm_layers=convlstm_layers,
            dropout=dropout,
            output_activation="none"  # Raw logits before head activations
        )

        # Dedicated 1x1 conv head for each horizon
        penultimate_ch = encoder_channels[0]
        self.heads = nn.ModuleDict({
            str(h): nn.Conv2d(penultimate_ch, 1, kernel_size=1)
            for h in self.horizons
        })

        if output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass producing predictions for all horizons simultaneously.

        Args:
            x: Input tensor [B, T=7, C=7, H, W] or [B, T*C, H, W].

        Returns:
            Dict mapping horizon h -> predicted SIC [B, 1, H, W].
        """
        if x.dim() == 4:
            B, TC, H, W = x.shape
            x = x.view(B, self.seq_len, self.in_channels, H, W)

        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        skip_connections = []
        cur = x_flat
        for encoder in self.backbone.encoders:
            cur, skip = encoder(cur)
            skip_connections.append(skip.view(B, T, skip.shape[1], skip.shape[2], skip.shape[3]))

        bottleneck_flat = self.backbone.bottleneck_conv(cur)
        _, C_bot, H_bot, W_bot = bottleneck_flat.shape
        bottleneck_seq = bottleneck_flat.view(B, T, C_bot, H_bot, W_bot)

        bottleneck_out, _ = self.backbone.temporal_bottleneck(bottleneck_seq)
        h_final = bottleneck_out[:, -1]
        h_refined = self.backbone.post_bottleneck_conv(h_final)

        dec = h_refined
        num_stages = len(self.backbone.decoders)
        for i, decoder in enumerate(self.backbone.decoders):
            skip_t = skip_connections[num_stages - 1 - i][:, -1]
            dec = decoder(dec, skip_t)

        # dec is [B, penultimate_ch, H, W]
        # Project through each horizon's head
        outputs = {}
        for h in self.horizons:
            raw_logits = self.heads[str(h)](dec)
            outputs[h] = self.activation(raw_logits)

        return outputs


class RecursiveMultiHorizonRollout(nn.Module):
    """
    Recursive Autoregressive Forecaster.

    Wraps any single-step model (e.g. UNetConvLSTM) and rolls it forward
    step-by-step up to max(horizons). At each step, the predicted SIC is fed back
    as channel 0 of the input sequence, with persistence assumed for environmental forcing.
    """

    def __init__(
        self,
        base_model: nn.Module,
        horizons: List[int] = [1, 3, 5, 7],
        seq_len: int = 7,
        in_channels: int = 7
    ):
        super().__init__()
        self.base_model = base_model
        self.horizons = sorted(horizons)
        self.max_horizon = max(self.horizons)
        self.seq_len = seq_len
        self.in_channels = in_channels

    def forward(
        self,
        x: torch.Tensor,
        future_forcing: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Autoregressive multi-step rollout.

        Args:
            x: Input history [B, T=7, C=7, H, W]
            future_forcing: Optional dict mapping step k -> known future forcing [B, C-1, H, W]

        Returns:
            Dict mapping horizon h -> predicted SIC [B, 1, H, W].
        """
        if x.dim() == 4:
            B, TC, H, W = x.shape
            x = x.view(B, self.seq_len, self.in_channels, H, W)

        cur_seq = x.clone()  # [B, T, C, H, W]
        outputs = {}

        for step in range(1, self.max_horizon + 1):
            # Predict next-day SIC [B, 1, H, W]
            pred_sic = self.base_model(cur_seq)

            if step in self.horizons:
                outputs[step] = pred_sic

            if step < self.max_horizon:
                # Construct next timestep array [B, C, H, W]
                # Channel 0: predicted SIC
                # Channels 1..6: environmental forcing (from future_forcing if given, else persistence from T-1)
                latest_forcing = cur_seq[:, -1, 1:].clone()
                if future_forcing and step in future_forcing:
                    latest_forcing = future_forcing[step]

                next_step_tensor = torch.cat([pred_sic, latest_forcing], dim=1)  # [B, C, H, W]

                # Roll sequence: drop oldest timestep t=0, append next_step_tensor at t=T
                cur_seq = torch.cat([cur_seq[:, 1:], next_step_tensor.unsqueeze(1)], dim=1)

        return outputs


class MultiHorizonForecaster(nn.Module):
    """
    Unified Multi-Horizon Forecaster supporting both 'direct' and 'recursive' modes.
    """

    def __init__(
        self,
        mode: str = "direct",
        horizons: List[int] = [1, 3, 5, 7],
        in_channels: int = 7,
        seq_len: int = 7,
        encoder_channels: List[int] = [32, 64, 128, 256],
        convlstm_layers: int = 1,
        dropout: float = 0.1,
        base_model: Optional[nn.Module] = None
    ):
        super().__init__()
        self.mode = mode.lower()
        self.horizons = sorted(horizons)
        self.in_channels = in_channels
        self.seq_len = seq_len

        if self.mode == "direct":
            self.model = DirectMultiHeadUNetConvLSTM(
                in_channels=in_channels,
                horizons=self.horizons,
                seq_len=seq_len,
                encoder_channels=encoder_channels,
                convlstm_layers=convlstm_layers,
                dropout=dropout
            )
        elif self.mode == "recursive":
            if base_model is None:
                base_model = UNetConvLSTM(
                    in_channels=in_channels,
                    output_channels=1,
                    seq_len=seq_len,
                    encoder_channels=encoder_channels,
                    convlstm_layers=convlstm_layers,
                    dropout=dropout
                )
            self.model = RecursiveMultiHorizonRollout(
                base_model=base_model,
                horizons=self.horizons,
                seq_len=seq_len,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'direct' or 'recursive'.")

    def forward(
        self,
        x: torch.Tensor,
        future_forcing: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, torch.Tensor]:
        if self.mode == "recursive":
            return self.model(x, future_forcing=future_forcing)
        return self.model(x)


class MultiHorizonLoss(nn.Module):
    """
    Weighted multi-horizon loss over ocean pixels.
    L = sum_{h} w_h * MaskedMAE(pred_h, target_h, mask)
    """

    def __init__(
        self,
        horizons: List[int] = [1, 3, 5, 7],
        weights: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        self.horizons = sorted(horizons)
        if weights is None:
            # Default: equal weighting across horizons
            self.weights = {h: 1.0 / len(self.horizons) for h in self.horizons}
        else:
            # Normalize weights to sum to 1.0
            total_w = sum(weights[h] for h in self.horizons)
            self.weights = {h: weights[h] / total_w for h in self.horizons}

        self.criterion = MaskedMAELoss()

    def forward(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: Dict[int, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute total weighted loss and individual per-horizon loss values.
        """
        total_loss = 0.0
        horizon_losses = {}

        for h in self.horizons:
            if h not in predictions or h not in targets:
                raise KeyError(f"Horizon {h} missing from predictions or targets.")

            pred_h = predictions[h]
            target_h = targets[h]

            loss_h = self.criterion(pred_h, target_h, mask)
            horizon_losses[h] = loss_h.item()
            total_loss = total_loss + self.weights[h] * loss_h

        return total_loss, horizon_losses


class MultiHorizonTrainer:
    """
    Production trainer for multi-horizon forecasting models.
    """

    def __init__(
        self,
        model: nn.Module,
        horizons: List[int] = [1, 3, 5, 7],
        mask: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        experiment_name: str = "multi_horizon_phase_d",
        device: str = "cpu"
    ):
        self.model = model
        self.horizons = sorted(horizons)
        self.device = torch.device(device)
        self.model.to(self.device)

        self.mask = mask.to(self.device) if mask is not None else None
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("models/checkpoints/phase_d")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.loss_fn = MultiHorizonLoss(horizons=self.horizons)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_step(
        self,
        x: torch.Tensor,
        targets: Dict[int, torch.Tensor]
    ) -> Tuple[float, Dict[int, float]]:
        """Run single training step on a batch."""
        self.model.train()
        self.optimizer.zero_grad()

        x = x.to(self.device)
        targets_dev = {h: targets[h].to(self.device) for h in self.horizons}

        preds = self.model(x)
        loss, h_losses = self.loss_fn(preds, targets_dev, self.mask)

        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        return loss.item(), h_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save atomic training checkpoint."""
        state = {
            "epoch": epoch,
            "experiment_name": self.experiment_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "horizons": self.horizons,
            "timestamp": datetime.utcnow().isoformat()
        }
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        tmp_path = latest_path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(latest_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(state, best_path)

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """Resume state from checkpoint file."""
        p = Path(checkpoint_path)
        checkpoint = torch.load(p, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        logger.info(f"Resumed from epoch {checkpoint['epoch']} with best loss {self.best_val_loss:.6f}")
        return self.start_epoch
