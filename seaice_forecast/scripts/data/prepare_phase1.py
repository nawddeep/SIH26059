#!/usr/bin/env python
"""
Prepare NSIDC SIC data for model training.

This script:
1. Creates land-ocean mask
2. Splits data into train/val/test
3. Creates sliding window sequences
4. Computes normalization statistics
5. Saves processed data

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config path/to/config.yaml
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths
from seaice_forecast.data_processing.preprocessing import SICPreprocessor
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SIC data for model training"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: use default config)"
    )
    parser.add_argument(
        "--skip-mask",
        action="store_true",
        help="Skip mask creation if it already exists"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()

    config = resolve_paths(config)

    logger.info("Starting data preparation...")
    logger.info(f"Configuration: {config['model']['name']}")

    # Create preprocessor
    preprocessor = SICPreprocessor(config)

    # Get all raw data files
    raw_dir = Path(config['data']['paths']['raw'])
    raw_files = sorted(raw_dir.glob("*.nc"))

    if len(raw_files) == 0:
        logger.error(f"No data files found in {raw_dir}")
        logger.error("Please run download script first: python scripts/download_data.py")
        return

    logger.info(f"Found {len(raw_files)} raw data files")

    # Step 1: Create or load land-ocean mask
    mask_path = Path(config['data']['paths']['processed']) / "land_ocean_mask.npy"

    if mask_path.exists() and args.skip_mask:
        logger.info("Loading existing land-ocean mask...")
        mask = preprocessor.load_mask(mask_path)
    else:
        logger.info("Creating land-ocean mask from sample files...")
        # Use 100 sample files distributed across the dataset
        step = max(1, len(raw_files) // 100)
        sample_files = raw_files[::step][:100]
        logger.info(f"Using {len(sample_files)} sample files for mask creation")
        mask = preprocessor.create_land_ocean_mask(sample_files, mask_path)

    ocean_pixels = np.sum(mask)
    total_pixels = mask.size
    logger.info(f"Mask summary: {ocean_pixels}/{total_pixels} ocean pixels "
               f"({ocean_pixels/total_pixels:.1%})")

    # Step 2: Split data chronologically
    logger.info("\nSplitting data into train/val/test sets...")
    splits = preprocessor.split_data(raw_files)

    print("\n" + "="*60)
    print("DATA SPLITS")
    print("="*60)
    for split_name, files in splits.items():
        if len(files) > 0:
            dates = [f.name.split('_')[4] for f in files[:1] + files[-1:]]
            print(f"{split_name.upper():5s}: {len(files):4d} files  "
                  f"({dates[0]} to {dates[-1] if len(dates) > 1 else dates[0]})")
        else:
            print(f"{split_name.upper():5s}: {len(files):4d} files")
    print("="*60)

    # Step 3: Process each split
    train_inputs = None

    for split_name, files in splits.items():
        if len(files) == 0:
            logger.warning(f"No files for {split_name}, skipping")
            continue

        logger.info(f"\nProcessing {split_name} split ({len(files)} files)...")

        try:
            inputs, targets, date_pairs = preprocessor.create_windows(files)
            preprocessor.save_processed_data(inputs, targets, date_pairs, split_name)

            # Save training inputs for statistics
            if split_name == 'train':
                train_inputs = inputs

        except Exception as e:
            logger.error(f"Error processing {split_name} split: {e}")
            raise

    # Step 4: Compute normalization statistics from training data
    if train_inputs is not None:
        logger.info("\nComputing normalization statistics from training data...")
        stats = preprocessor.compute_statistics(train_inputs)

        print("\n" + "="*60)
        print("NORMALIZATION STATISTICS (computed from training set)")
        print("="*60)
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Std:  {stats['std']:.6f}")
        print("="*60)

    # Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Processed data saved to: {config['data']['paths']['processed']}")
    print(f"Land-ocean mask: {mask_path}")
    print("\nNext steps:")
    print("  1. Evaluate baselines: python scripts/evaluate_baselines.py")
    print("  2. Train model: python scripts/train.py")
    print("="*60)


if __name__ == "__main__":
    main()
