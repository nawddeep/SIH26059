#!/usr/bin/env python3
"""
Prepare training data from regridded NPZ files.

This script creates sliding windows from the regridded daily files for training.
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parameters
INPUT_WINDOW = 7  # Days of history
FORECAST_HORIZON = 1  # Predict next day
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"

def parse_date_from_filename(filepath):
    """Extract date from filename like 20130416.npz"""
    date_str = filepath.stem  # e.g., "20130416"
    return datetime.strptime(date_str, "%Y%m%d")

def discover_files(regridded_dir):
    """Find all regridded NPZ files sorted by date"""
    files = []
    for year_dir in sorted(regridded_dir.glob("*")):
        if year_dir.is_dir():
            for npz_file in year_dir.glob("*.npz"):
                try:
                    date = parse_date_from_filename(npz_file)
                    files.append((date, npz_file))
                except:
                    continue
    
    files.sort(key=lambda x: x[0])
    return files

def split_files_by_date(files):
    """Split files chronologically"""
    train_end = datetime.strptime(TRAIN_END, "%Y-%m-%d")
    val_end = datetime.strptime(VAL_END, "%Y-%m-%d")
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for date, filepath in files:
        if date <= train_end:
            splits['train'].append((date, filepath))
        elif date <= val_end:
            splits['val'].append((date, filepath))
        else:
            splits['test'].append((date, filepath))
    
    return splits

def create_windows(files, var_index=0):
    """
    Create sliding windows from files.
    var_index=0 for SIC (first variable)
    Returns: (inputs, targets, dates)
    """
    # Load all data into memory
    logger.info(f"Loading {len(files)} files...")
    data_list = []
    dates = []
    
    for date, filepath in tqdm(files):
        try:
            npz = np.load(filepath)
            # Extract just SIC (first variable)
            sic = npz['data'][var_index, :, :]  # Shape: (332, 316)
            data_list.append(sic)
            dates.append(date)
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")
            continue
    
    if len(data_list) < INPUT_WINDOW + FORECAST_HORIZON:
        raise ValueError(f"Not enough data: {len(data_list)} files")
    
    # Stack into array
    data = np.stack(data_list, axis=0)  # (N, 332, 316)
    logger.info(f"Data shape: {data.shape}")
    
    # Create sliding windows
    inputs = []
    targets = []
    date_pairs = []
    
    for i in range(len(data) - INPUT_WINDOW - FORECAST_HORIZON + 1):
        input_window = data[i:i+INPUT_WINDOW]  # (7, 332, 316)
        target = data[i+INPUT_WINDOW:i+INPUT_WINDOW+FORECAST_HORIZON]  # (1, 332, 316)
        
        inputs.append(input_window)
        targets.append(target)
        date_pairs.append({
            'input_start': dates[i].strftime('%Y-%m-%d'),
            'input_end': dates[i+INPUT_WINDOW-1].strftime('%Y-%m-%d'),
            'target': dates[i+INPUT_WINDOW].strftime('%Y-%m-%d')
        })
    
    inputs = np.stack(inputs, axis=0)  # (N_windows, 7, 332, 316)
    targets = np.stack(targets, axis=0)  # (N_windows, 1, 332, 316)
    
    logger.info(f"Created {len(inputs)} windows")
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    return inputs, targets, date_pairs

def create_mask(data_sample):
    """Create land/ocean mask from data"""
    # Land pixels have consistent NaN or 255 values
    # Ocean pixels have valid SIC values [0, 1]
    
    # A pixel is ocean if it ever has a valid value in [0, 1]
    mask = np.zeros(data_sample.shape[-2:], dtype=bool)
    
    # Check multiple samples
    for i in range(min(100, len(data_sample))):
        sample = data_sample[i, 0, :, :]  # First day of window
        valid = np.isfinite(sample) & (sample >= 0) & (sample <= 1.5)
        mask |= valid
    
    return mask.astype(np.uint8)

def compute_statistics(train_inputs, mask):
    """Compute mean and std from training data (ocean pixels only)"""
    # Flatten all training data
    ocean_values = train_inputs[:, :, mask == 1].flatten()
    
    # Remove any remaining NaN/inf
    ocean_values = ocean_values[np.isfinite(ocean_values)]
    
    mean = float(np.mean(ocean_values))
    std = float(np.std(ocean_values))
    
    return {'mean': mean, 'std': std}

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    regridded_dir = base_dir / "data" / "processed" / "regridded" / "daily"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PREPARING TRAINING DATA FROM REGRIDDED FILES")
    logger.info("="*60)
    
    # Discover files
    logger.info(f"\nScanning {regridded_dir}...")
    files = discover_files(regridded_dir)
    logger.info(f"Found {len(files)} regridded files")
    
    if len(files) == 0:
        logger.error("No regridded files found!")
        return
    
    logger.info(f"Date range: {files[0][0].date()} to {files[-1][0].date()}")
    
    # Split chronologically
    logger.info("\nSplitting data chronologically...")
    splits = split_files_by_date(files)
    
    print("\n" + "="*60)
    print("DATA SPLITS")
    print("="*60)
    for split_name, split_files in splits.items():
        if len(split_files) > 0:
            start_date = split_files[0][0].date()
            end_date = split_files[-1][0].date()
            print(f"{split_name.upper():5s}: {len(split_files):4d} files  "
                  f"({start_date} to {end_date})")
        else:
            print(f"{split_name.upper():5s}: {len(split_files):4d} files")
    print("="*60)
    
    # Process each split
    train_inputs = None
    mask = None
    
    for split_name, split_files in splits.items():
        if len(split_files) < INPUT_WINDOW + FORECAST_HORIZON:
            logger.warning(f"Not enough files for {split_name} split, skipping")
            continue
        
        logger.info(f"\nProcessing {split_name} split...")
        
        inputs, targets, date_pairs = create_windows(split_files)
        
        # Save data
        output_path = processed_dir / f"{split_name}_data.npz"
        logger.info(f"Saving to {output_path}...")
        np.savez_compressed(
            output_path,
            inputs=inputs,
            targets=targets
        )
        
        # Save metadata
        metadata_path = processed_dir / f"{split_name}_metadata.json"
        logger.info(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, 'w') as f:
            json.dump({
                'num_samples': len(inputs),
                'input_shape': list(inputs.shape),
                'target_shape': list(targets.shape),
                'date_pairs': date_pairs
            }, f, indent=2)
        
        # Save training inputs for mask and stats
        if split_name == 'train':
            train_inputs = inputs
    
    # Create mask from training data
    if train_inputs is not None:
        logger.info("\nCreating land/ocean mask...")
        mask = create_mask(train_inputs)
        ocean_pixels = np.sum(mask)
        total_pixels = mask.size
        logger.info(f"Mask: {ocean_pixels}/{total_pixels} ocean pixels "
                   f"({ocean_pixels/total_pixels:.1%})")
        
        mask_path = processed_dir / "land_ocean_mask.npy"
        np.save(mask_path, mask)
        logger.info(f"Saved mask to {mask_path}")
        
        # Compute statistics
        logger.info("\nComputing normalization statistics...")
        stats = compute_statistics(train_inputs, mask)
        
        print("\n" + "="*60)
        print("NORMALIZATION STATISTICS")
        print("="*60)
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Std:  {stats['std']:.6f}")
        print("="*60)
        
        stats_path = processed_dir / "normalization_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {processed_dir}/train_data.npz")
    print(f"  - {processed_dir}/val_data.npz")
    print(f"  - {processed_dir}/test_data.npz")
    print(f"  - {processed_dir}/land_ocean_mask.npy")
    print(f"  - {processed_dir}/normalization_stats.json")
    print("\nYou can now run training!")
    print("="*60)

if __name__ == '__main__':
    main()
