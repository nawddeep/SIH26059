"""
Training pipeline for U-Net model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskedMAELoss(nn.Module):
    """
    Masked Mean Absolute Error loss.

    Only computes loss over ocean pixels (mask=1), excluding land pixels.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked MAE.

        Args:
            predictions: Predicted SIC [B, 1, H, W]
            targets: Target SIC [B, 1, H, W]
            mask: Land-ocean mask [B, H, W] or [H, W], 1=ocean, 0=land

        Returns:
            Scalar loss
        """
        # Ensure mask has the right shape
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        # Broadcast mask to match predictions
        mask = mask.expand_as(predictions)

        # Compute absolute errors
        errors = torch.abs(predictions - targets)

        # Apply mask (only ocean pixels)
        masked_errors = errors * mask

        # Compute mean over ocean pixels only
        n_ocean_pixels = mask.sum()
        if n_ocean_pixels > 0:
            loss = masked_errors.sum() / n_ocean_pixels
        else:
            loss = masked_errors.sum()  # Fallback if no ocean pixels

        return loss


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimization, 'max' for maximization
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.comparator = np.less if mode == 'min' else np.greater

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Check if there's improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class Trainer:
    """
    Trainer for U-Net model.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        mask: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.

        Args:
            model: U-Net model
            config: Configuration dictionary
            mask: Land-ocean mask
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(self.device)

        # Convert mask to tensor and move to device
        if mask is not None:
            self.mask = torch.from_numpy(mask).float().to(self.device)
        else:
            self.mask = None

        # Loss function
        self.criterion = MaskedMAELoss()

        # Optimizer
        lr = config['training']['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        if config['training']['lr_scheduler']['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config['training']['lr_scheduler']['factor'],
                patience=config['training']['lr_scheduler']['patience'],
                min_lr=config['training']['lr_scheduler']['min_lr']
            )
        else:
            self.scheduler = None

        # Early stopping
        early_stop_config = config['training']['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
            mode='min'
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets, masks in dataloader:
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Use provided mask or default mask
            if self.mask is not None:
                batch_mask = self.mask
            else:
                batch_mask = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            loss = self.criterion(outputs, targets, batch_mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate model.

        Args:
            dataloader: Validation dataloader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, targets, masks in dataloader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Use provided mask or default mask
                if self.mask is not None:
                    batch_mask = self.mask
                else:
                    batch_mask = masks.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets, batch_mask)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        save_dir: str
    ) -> Dict:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            n_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Model: {self.config['model']['name']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {n_epochs}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")

        # Training loop
        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss = self.validate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)

            # Log progress
            logger.info(
                f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                best_model_path = save_path / f"{self.config['model']['name']}_best.pt"
                self.save_checkpoint(best_model_path, epoch + 1, val_loss, is_best=True)
                logger.info(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

            # Check early stopping
            if self.early_stopping(val_loss):
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                logger.info(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
                break

        # Save final model
        final_model_path = save_path / f"{self.config['model']['name']}_final.pt"
        self.save_checkpoint(final_model_path, epoch + 1, val_loss, is_best=False)

        # Save training history
        history_path = save_path / f"{self.config['model']['name']}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best model: {best_model_path}")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"Training history: {history_path}")

        return self.history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"  Epoch: {checkpoint['epoch']}")
        logger.info(f"  Val loss: {checkpoint['val_loss']:.6f}")


def main():
    """Test trainer."""
    from ..models.unet import UNet

    # Create dummy model
    model = UNet(input_channels=7, output_channels=1, encoder_channels=[16, 32])

    # Dummy config
    config = {
        'model': {'name': 'test_model'},
        'training': {
            'batch_size': 2,
            'learning_rate': 0.001,
            'lr_scheduler': {
                'type': 'reduce_on_plateau',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6
            },
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.0001
            }
        }
    }

    # Create trainer
    trainer = Trainer(model, config, device='cpu')

    print("Trainer initialized successfully")
    print(f"Device: {trainer.device}")
    print(f"Loss function: {trainer.criterion.__class__.__name__}")


if __name__ == "__main__":
    main()
