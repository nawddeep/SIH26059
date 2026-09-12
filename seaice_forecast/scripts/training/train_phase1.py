#!/usr/bin/env python
"""
Train U-Net model for sea-ice concentration forecasting.

This script trains the U-Net model using the processed data and saves
checkpoints during training.

Usage:
    python scripts/train.py
    python scripts/train.py --config path/to/config.yaml --epochs 50
    python scripts/train.py --device cpu
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths
from seaice_forecast.models.unet import create_unet_from_config
from seaice_forecast.data_processing.dataset_phase1 import create_dataloaders
from seaice_forecast.training.trainer import Trainer
from seaice_forecast.utils.visualization import plot_training_history
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net model for SIC forecasting"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: use default config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to train on: cuda or cpu (overrides config)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint to resume training from"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    config = resolve_paths(config)

    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device:
        config['compute']['device'] = args.device

    # Set device
    device = config['compute']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            logger.info("CUDA not available, using Apple Silicon MPS device")
            device = 'mps'
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = 'cpu'

    logger.info("="*80)
    logger.info("SEA-ICE CONCENTRATION FORECASTING - MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")

    # Check if processed data exists
    processed_dir = Path(config['data']['paths']['processed'])
    train_data = processed_dir / "train_data.npz"
    val_data = processed_dir / "val_data.npz"

    if not train_data.exists() or not val_data.exists():
        logger.error("Processed data not found!")
        logger.error("Please run: python scripts/prepare_data.py")
        return

    # Load land-ocean mask
    mask_file = processed_dir / "land_ocean_mask.npy"
    if mask_file.exists():
        mask = np.load(mask_file)
        logger.info(f"Loaded mask: {mask.shape}, ocean pixels: {np.sum(mask)}")
    else:
        logger.warning("No mask found, training on all pixels")
        mask = None

    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    dataloaders = create_dataloaders(config, mask)

    if 'train' not in dataloaders or 'val' not in dataloaders:
        logger.error("Failed to create train and validation dataloaders")
        return

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    logger.info(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Create model
    logger.info("\nCreating U-Net model...")
    model = create_unet_from_config(config)

    # Log model info
    params = model.get_num_parameters()
    logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    logger.info(f"Model architecture:")
    logger.info(f"  Input: [B, {config['forecast']['input_window']}, H, W]")
    logger.info(f"  Output: [B, {config['architecture']['output_shape'][0]}, H, W]")
    logger.info(f"  Encoder channels: {config['architecture']['encoder_channels'][1:]}")

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        mask=mask,
        device=device
    )

    # Load checkpoint if resuming
    if args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train model
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)

    output_dir = Path(config['output']['model_dir'])

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config['training']['epochs'],
            save_dir=str(output_dir)
        )

        # Plot training history
        logger.info("\nGenerating training plots...")
        plots_dir = Path(config['output']['plots_dir'])
        plots_dir.mkdir(parents=True, exist_ok=True)

        history_file = output_dir / f"{config['model']['name']}_history.json"
        history_plot = plots_dir / f"{config['model']['name']}_training_history.png"

        plot_training_history(str(history_file), str(history_plot))
        logger.info(f"Training history plot saved to {history_plot}")

        # Summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Model: {config['model']['name']}")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"Best epoch: {trainer.best_epoch}")
        model_name = config['model']['name']
        print(f"\nSaved files:")
        print(f"  Best model: {output_dir / (model_name + '_best.pt')}")
        print(f"  Final model: {output_dir / (model_name + '_final.pt')}")
        print(f"  History: {history_file}")
        print(f"  Training plot: {history_plot}")
        print("\n" + "-"*80)
        print("NEXT STEP:")
        print("-"*80)
        print("Evaluate the trained model:")
        print(f"  python scripts/evaluate_model.py --checkpoint {output_dir / (model_name + '_best.pt')}")
        print("="*80)

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info("Partial results have been saved")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
