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
import platform
import os
import subprocess
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# Inline Hardware Detection and Auto-Tuning (No external dependencies)
# =============================================================================

def detect_hardware():
    """Detect Mac hardware and print configuration."""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        chip = platform.processor() or "Unknown CPU"
    
    cores = os.cpu_count() or 1
    
    try:
        mem_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip())
        mem_gb = mem_bytes / (1024**3)
    except Exception:
        mem_gb = 8.0  # fallback
    
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    cuda_available = torch.cuda.is_available()
    
    print("=" * 70)
    print("HARDWARE DETECTION")
    print("=" * 70)
    print(f"Chip:      {chip}")
    print(f"Cores:     {cores}")
    print(f"RAM:       {mem_gb:.1f} GB")
    print(f"MPS:       {mps_available}")
    print(f"CUDA:      {cuda_available}")
    print(f"PyTorch:   {torch.__version__}")
    print("=" * 70)
    
    return {
        "chip": chip,
        "cores": cores,
        "mem_gb": mem_gb,
        "mps": mps_available,
        "cuda": cuda_available
    }


def quick_batch_size_probe(model, sample_input_shape, device, candidate_sizes=[2, 4, 8, 16]):
    """
    Quick batch size calibration (~15-20 seconds).
    Tests a few candidates and picks the largest that runs cleanly.
    
    Args:
        model: The model to test
        sample_input_shape: Tuple of (C, H, W) for input
        device: torch device
        candidate_sizes: List of batch sizes to test
        
    Returns:
        Best batch size (int)
    """
    print("\nQuick batch size probe (testing a few candidates)...")
    model.eval()
    
    best_size = candidate_sizes[0]
    best_throughput = 0.0
    
    for batch_size in candidate_sizes:
        try:
            # Create dummy batch
            dummy_input = torch.randn(batch_size, *sample_input_shape, device=device)
            
            # Warmup
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Time a few iterations
            torch.mps.synchronize() if device.type == "mps" else None
            start_time = time.time()
            n_iters = 3
            
            with torch.no_grad():
                for _ in range(n_iters):
                    _ = model(dummy_input)
            
            torch.mps.synchronize() if device.type == "mps" else None
            elapsed = time.time() - start_time
            
            samples_per_sec = (batch_size * n_iters) / elapsed
            
            print(f"  Batch size {batch_size:2d}: {samples_per_sec:.1f} samples/sec - OK")
            
            if samples_per_sec > best_throughput:
                best_throughput = samples_per_sec
                best_size = batch_size
            
            # Clean up
            del dummy_input
            torch.mps.empty_cache() if device.type == "mps" else None
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                print(f"  Batch size {batch_size:2d}: Memory allocation failed - stopping")
                break
            else:
                print(f"  Batch size {batch_size:2d}: Error ({str(e)[:50]}) - stopping")
                break
        except Exception as e:
            print(f"  Batch size {batch_size:2d}: Unexpected error - stopping")
            break
    
    print(f"\nSelected batch size: {best_size} ({best_throughput:.1f} samples/sec)")
    return best_size


def quick_worker_probe(dataset, batch_size, worker_candidates=[0, 2]):
    """
    Quick worker count check.
    Tests 0 vs 2 workers, picks faster. Falls back to 0 on errors.
    
    Args:
        dataset: Dataset to test
        batch_size: Batch size to use
        worker_candidates: Worker counts to test
        
    Returns:
        Best num_workers (int)
    """
    print("\nQuick worker count probe...")
    
    best_workers = 0
    best_time = float('inf')
    
    for num_workers in worker_candidates:
        try:
            from torch.utils.data import DataLoader
            
            # Test small number of batches
            n_batches = min(5, len(dataset) // batch_size)
            if n_batches == 0:
                print(f"  Workers {num_workers}: Not enough data for testing")
                continue
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(num_workers > 0),
                persistent_workers=(num_workers > 0),
                drop_last=False,
                timeout=10 if num_workers > 0 else 0
            )
            
            start_time = time.time()
            batches_loaded = 0
            
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                batches_loaded += 1
            
            elapsed = time.time() - start_time
            
            if batches_loaded > 0:
                time_per_batch = elapsed / batches_loaded
                print(f"  Workers {num_workers}: {time_per_batch*1000:.1f} ms/batch - OK")
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_workers = num_workers
            
        except Exception as e:
            print(f"  Workers {num_workers}: Error ({str(e)[:50]}) - skipping")
            continue
    
    print(f"\nSelected num_workers: {best_workers}")
    return best_workers


def auto_select_device(hw_info):
    """Select best device based on hardware detection."""
    if hw_info["mps"]:
        device = torch.device("mps")
        print(f"\n[Device] Using: MPS (Apple Silicon GPU)")
    elif hw_info["cuda"]:
        device = torch.device("cuda")
        print(f"\n[Device] Using: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n[Device] Using: CPU (no GPU acceleration)")
    
    return device

from seaice_forecast.config import load_config, resolve_paths, get_project_root
from seaice_forecast.models.unet import UNet, create_unet_from_config
from seaice_forecast.data_processing.dataset_phase2 import create_phase2_dataloaders
from seaice_forecast.training.trainer import Trainer
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
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['compute']['device'] = args.device
    
    # =============================================================================
    # AUTO-DETECTION: Hardware, Device, Batch Size, Workers
    # =============================================================================
    
    print("\n")
    hw_info = detect_hardware()
    
    # Auto-select device (or use command-line override)
    if args.device:
        device = torch.device(args.device)
        print(f"\n[Device] Using command-line override: {device}")
    else:
        device = auto_select_device(hw_info)

    logger.info("="*80)
    logger.info("PHASE 2: ENVIRONMENTAL U-NET TRAINING")
    logger.info("="*80)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Phase: {config['model']['phase']}")
    logger.info(f"Input: {config['architecture']['input_description']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
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
    
    # =============================================================================
    # AUTO-TUNE: Batch size and num_workers
    # =============================================================================
    
    # Move model to device for batch size probing
    model.to(device)
    
    # Phase 2: 49 channels (7 days × 7 variables) - larger memory footprint than Phase 1
    sample_h, sample_w = 316, 332  # Antarctic grid size
    
    # Quick batch size probe (unless explicitly overridden by command line)
    if args.batch_size:
        optimized_batch_size = args.batch_size
        print(f"\n[Batch Size] Using command-line override: {optimized_batch_size}")
    else:
        print("\n[Phase 2 Note] Testing batch sizes with 49-channel input (larger than Phase 1's 7 channels)")
        optimized_batch_size = quick_batch_size_probe(
            model=model,
            sample_input_shape=(49, sample_h, sample_w),
            device=device,
            candidate_sizes=[2, 4, 8, 16]  # May settle lower than Phase 1 due to 7x more channels
        )
    
    # Update config with optimized batch size
    config['training']['batch_size'] = optimized_batch_size
    
    # Create Phase 2 dataloaders
    logger.info("\nCreating Phase 2 dataloaders...")
    dataloaders = create_phase2_dataloaders(config, mask, data_dir)

    if 'train' not in dataloaders or 'val' not in dataloaders:
        logger.error("Failed to create train and validation dataloaders")
        return

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    logger.info(f"\nTrain: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    
    # Show channel organization
    sample_dataset = train_loader.dataset
    if hasattr(sample_dataset, 'get_channel_info'):
        channel_info = sample_dataset.get_channel_info()
        logger.info("\nChannel organization:")
        for var, info in channel_info.items():
            logger.info(f"  {var}: channels {info['start']}-{info['end']-1}")
    
    # Quick worker count probe on training dataset
    optimized_num_workers = quick_worker_probe(
        dataset=train_loader.dataset,
        batch_size=optimized_batch_size,
        worker_candidates=[0, 2]
    )
    
    # Recreate dataloaders with optimized worker count
    if optimized_num_workers != 0:
        logger.info(f"\nRecreating dataloaders with optimized num_workers={optimized_num_workers}...")
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=optimized_batch_size,
            shuffle=True,
            num_workers=optimized_num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=optimized_batch_size,
            shuffle=False,
            num_workers=optimized_num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )
    
    # =============================================================================
    # Print final optimized settings
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("OPTIMIZED TRAINING SETTINGS (PHASE 2: 49 CHANNELS)")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Batch Size:       {optimized_batch_size}")
    print(f"Num Workers:      {optimized_num_workers}")
    print(f"Precision:        fp32 (stable default)")
    print("=" * 70)

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
