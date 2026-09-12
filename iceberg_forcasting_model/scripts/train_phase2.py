#!/usr/bin/env python
"""
Phase 2: Train Environmental U-Net for sea-ice concentration forecasting.

This script trains the U-Net with multi-variable environmental forcing:
- Input: 49 channels (7 days × 7 variables)
- Variables: SIC, wind_u, wind_v, air_temp, sst, current_u, current_v
- Output: Next-day SIC forecast
- Model version: sic_unet_env_v001

Key differences from Phase 1:
- Uses Phase2Dataset with 49-channel input
- Model initialized with input_channels=49
- Otherwise preserves Phase 1 architecture and training setup

Usage:
    python scripts/train_phase2.py
    python scripts/train_phase2.py --epochs 100 --batch-size 8
    python scripts/train_phase2.py --checkpoint models/sic_unet_env_v001_checkpoint.pt
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths, get_project_root
from seaice_forecast.models.unet import UNet, create_unet_from_config
from seaice_forecast.data_processing.dataset_phase2 import create_phase2_dataloaders
from seaice_forecast.evaluation.trainer import Trainer
from seaice_forecast.utils.visualization import plot_training_history
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_phase2_config(base_config: dict) -> dict:
    """
    Create Phase 2 configuration from base config.

    Modifies config for multi-variable input while preserving
    Phase 1 hyperparameters for fair comparison.

    Args:
        base_config: Phase 1 configuration

    Returns:
        Phase 2 configuration
    """
    config = base_config.copy()

    # Update model metadata
    config['model'] = {
        'name': 'sic_unet_env_v001',
        'version': '0.2.0',
        'phase': 2,
        'description': 'Environmental U-Net with multi-variable forcing'
    }

    # Update architecture input shape for documentation
    # Actual input channels handled in model creation
    config['architecture']['input_shape'] = [49, 316, 332]
    config['architecture']['input_description'] = (
        '49 channels = 7 days × 7 variables '
        '(SIC, wind_u, wind_v, air_temp, sst, current_u, current_v)'
    )

    # Document environmental variables
    config['environmental_forcing'] = {
        'enabled': True,
        'variables': [
            'wind_u',
            'wind_v',
            'air_temp',
            'sst',
            'current_u',
            'current_v'
        ],
        'n_variables': 7,  # Including SIC
        'channels_per_variable': 7,
        'total_channels': 49
    }

    # Keep Phase 1 hyperparameters unchanged for fair comparison
    # Same batch size, learning rate, optimizer, loss function

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train Phase 2 Environmental U-Net for SIC forecasting"
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
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
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
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with Phase 2 data (default: data/processed/phase2)"
    )

    args = parser.parse_args()

    # Load base configuration
    if args.config:
        base_config = load_config(args.config)
    else:
        base_config = load_config()

    base_config = resolve_paths(base_config)

    # Create Phase 2 config
    config = create_phase2_config(base_config)

    # Override config with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
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
    logger.info("PHASE 2: ENVIRONMENTAL U-NET TRAINING")
    logger.info("="*80)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Phase: {config['model']['phase']}")
    logger.info(f"Input: {config['architecture']['input_description']}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info("")
    logger.info("Environmental variables:")
    for var in config['environmental_forcing']['variables']:
        logger.info(f"  - {var}")

    # Setup directories
    project_root = get_project_root()
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / 'data' / 'processed' / 'phase2'

    # Check if Phase 2 data exists
    train_data = data_dir / "train_data_phase2.npz"
    val_data = data_dir / "val_data_phase2.npz"

    if not train_data.exists() or not val_data.exists():
        logger.error("")
        logger.error("="*80)
        logger.error("PHASE 2 DATA NOT FOUND")
        logger.error("="*80)
        logger.error(f"Expected location: {data_dir}")
        logger.error("")
        logger.error("Please run Phase 2 data preparation first:")
        logger.error("  python scripts/prepare_environmental_data.py --download --regrid")
        logger.error("")
        logger.error("This will:")
        logger.error("  1. Download ERA5 variables (wind, temperature, SST)")
        logger.error("  2. Download Copernicus Marine ocean currents")
        logger.error("  3. Regrid all to SIC grid")
        logger.error("  4. Compute normalization statistics")
        logger.error("  5. Create combined multi-variable datasets")
        logger.error("="*80)
        return

    # Load land-ocean mask
    processed_dir = project_root / 'data' / 'processed'
    mask_file = processed_dir / "land_ocean_mask.npy"

    if mask_file.exists():
        mask = np.load(mask_file)
        logger.info(f"\nLoaded mask: {mask.shape}, ocean pixels: {np.sum(mask)}")
    else:
        logger.warning("No mask found, training on all pixels")
        mask = None

    # Create Phase 2 dataloaders
    logger.info("\nCreating Phase 2 dataloaders...")
    dataloaders = create_phase2_dataloaders(config, mask, data_dir)

    if 'train' not in dataloaders or 'val' not in dataloaders:
        logger.error("Failed to create train and validation dataloaders")
        return

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    logger.info(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Show channel organization
    sample_dataset = train_loader.dataset
    if hasattr(sample_dataset, 'get_channel_info'):
        channel_info = sample_dataset.get_channel_info()
        logger.info("\nChannel organization:")
        for var, info in channel_info.items():
            logger.info(f"  {var}: channels {info['start']}-{info['end']-1}")

    # Create model with 49 input channels
    logger.info("\nCreating Environmental U-Net model...")
    logger.info("Architecture: Same as Phase 1 (encoder-decoder with skip connections)")
    logger.info("Key difference: Input channels 7 → 49")

    # Explicitly create model with 49 channels
    model = UNet(
        input_channels=49,
        output_channels=1,
        encoder_channels=config['architecture']['encoder_channels'][1:],
        use_batch_norm=config['architecture']['use_batch_norm'],
        dropout=0.0,
        output_activation=config['architecture']['output_activation']
    )

    # Log model info
    params = model.get_num_parameters()
    logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    logger.info(f"  Input: [B, 49, H, W]")
    logger.info(f"  Output: [B, 1, H, W]")
    logger.info(f"  Encoder channels: {config['architecture']['encoder_channels'][1:]}")

    # Check parameter increase vs Phase 1
    # Phase 1 had ~2.3M parameters with 7 input channels
    # Phase 2 should have more in first layer due to 49 channels
    logger.info(f"\nParameter comparison to Phase 1:")
    logger.info(f"  Phase 1 (7 channels):  ~2,300,000 parameters")
    logger.info(f"  Phase 2 (49 channels): {params['total']:,} parameters")
    logger.info(f"  Increase: {params['total'] - 2_300_000:,} parameters")
    logger.info(f"  (Increase primarily in first conv layer: 7→49 channels)")

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
    logger.info("Same training procedure as Phase 1:")
    logger.info("  - Masked MAE loss (land pixels excluded)")
    logger.info("  - Adam optimizer")
    logger.info("  - ReduceLROnPlateau scheduler")
    logger.info("  - Early stopping with patience 15")
    logger.info("")

    output_dir = Path(config['output']['model_dir'])

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config['training']['epochs'],
            save_dir=str(output_dir)
        )

        # Save Phase 2 specific metadata
        phase2_metadata = {
            'model_name': config['model']['name'],
            'phase': 2,
            'training_completed': datetime.now().isoformat(),
            'environmental_forcing': config['environmental_forcing'],
            'input_channels': 49,
            'variable_order': ['sic', 'wind_u', 'wind_v', 'air_temp', 'sst', 'current_u', 'current_v'],
            'data_sources': {
                'sic': 'NOAA/NSIDC CDR v6',
                'wind': 'ERA5',
                'air_temp': 'ERA5',
                'sst': 'ERA5',
                'currents': 'Copernicus Marine GLORYS12V1'
            },
            'best_val_loss': float(trainer.best_val_loss),
            'best_epoch': int(trainer.best_epoch),
            'total_parameters': params['total']
        }

        metadata_file = output_dir / f"{config['model']['name']}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(phase2_metadata, f, indent=2)

        logger.info(f"\nPhase 2 metadata saved to {metadata_file}")

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
        print("PHASE 2 TRAINING COMPLETE")
        print("="*80)
        print(f"Model: {config['model']['name']}")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"Best epoch: {trainer.best_epoch}")
        model_name = config['model']['name']
        print(f"\nSaved files:")
        print(f"  Best model: {output_dir / (model_name + '_best.pt')}")
        print(f"  Final model: {output_dir / (model_name + '_final.pt')}")
        print(f"  Metadata: {metadata_file}")
        print(f"  History: {history_file}")
        print(f"  Training plot: {history_plot}")
        print("\n" + "-"*80)
        print("NEXT STEPS:")
        print("-"*80)
        print("1. Evaluate Phase 2 model:")
        print(f"   python scripts/evaluate_phase2.py --checkpoint {output_dir / (model_name + '_best.pt')}")
        print("")
        print("2. Compare Phase 1 vs Phase 2 vs baselines:")
        print("   python scripts/compare_phase1_phase2.py")
        print("")
        print("3. Run ablation studies:")
        print("   python scripts/ablation_analysis.py")
        print("="*80)

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info("Partial results have been saved")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
