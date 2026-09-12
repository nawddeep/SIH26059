#!/usr/bin/env python
"""
Create synthetic demo data for testing the Phase 1 workflow.

This generates realistic-looking SIC data without requiring NSIDC downloads.
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths

def generate_synthetic_sic(H=316, W=332, seed=42):
    """Generate realistic synthetic SIC field."""
    np.random.seed(seed)
    
    # Create base pattern (radial from pole)
    y, x = np.ogrid[:H, :W]
    center_y, center_x = H // 2, W // 2
    
    # Distance from center (pole)
    distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_distance = np.sqrt(center_y**2 + center_x**2)
    
    # SIC decreases with distance from pole
    base_sic = 1.0 - (distance / max_distance) ** 1.5
    base_sic = np.clip(base_sic, 0, 1)
    
    # Add some randomness
    noise = np.random.randn(H, W) * 0.1
    sic = np.clip(base_sic + noise, 0, 1)
    
    return sic

def create_land_ocean_mask(H=316, W=332):
    """Create simple land-ocean mask."""
    mask = np.ones((H, W))
    
    # Make outer edge "land" (Antarctica continent)
    mask[:20, :] = 0
    mask[-20:, :] = 0
    mask[:, :20] = 0
    mask[:, -20:] = 0
    
    return mask

def create_demo_dataset():
    """Create complete demo dataset."""
    print("Creating synthetic demo data...")
    
    # Load config
    config = load_config()
    config = resolve_paths(config)
    
    processed_dir = Path(config['data']['paths']['processed'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameters
    H, W = 316, 332
    input_window = 7
    forecast_horizon = 1
    
    # Create land-ocean mask
    print("Creating land-ocean mask...")
    mask = create_land_ocean_mask(H, W)
    np.save(processed_dir / "land_ocean_mask.npy", mask)
    print(f"  Mask: {mask.shape}, ocean pixels: {np.sum(mask)}")
    
    # Generate datasets
    splits = {
        'train': 365,  # 1 year
        'val': 120,    # ~4 months
        'test': 120    # ~4 months
    }
    
    for split_name, n_days in splits.items():
        print(f"\nCreating {split_name} split ({n_days} days)...")
        
        # Generate time series
        sic_sequence = []
        base_date = datetime(2020, 1, 1)
        
        for day in range(n_days):
            # Add temporal variation
            seed = day + (100 if split_name == 'val' else 200 if split_name == 'test' else 0)
            sic = generate_synthetic_sic(H, W, seed=seed)
            
            # Add seasonal trend
            seasonal_factor = 0.2 * np.sin(2 * np.pi * day / 365)
            sic = np.clip(sic + seasonal_factor, 0, 1)
            
            sic_sequence.append(sic)
        
        sic_sequence = np.array(sic_sequence)
        
        # Create windows
        inputs = []
        targets = []
        date_pairs = []
        
        for i in range(len(sic_sequence) - input_window - forecast_horizon + 1):
            input_seq = sic_sequence[i:i + input_window]
            target = sic_sequence[i + input_window:i + input_window + forecast_horizon]
            
            inputs.append(input_seq)
            targets.append(target)
            
            input_end_date = base_date + timedelta(days=i + input_window - 1)
            target_date = base_date + timedelta(days=i + input_window + forecast_horizon - 1)
            date_pairs.append((
                input_end_date.strftime("%Y-%m-%d"),
                target_date.strftime("%Y-%m-%d")
            ))
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        print(f"  Inputs: {inputs.shape}, Targets: {targets.shape}")
        print(f"  Samples: {len(inputs)}")
        
        # Save data
        data_file = processed_dir / f"{split_name}_data.npz"
        np.savez_compressed(data_file, inputs=inputs, targets=targets)
        
        # Save metadata
        metadata = {
            'split': split_name,
            'n_samples': len(inputs),
            'input_shape': list(inputs.shape),
            'target_shape': list(targets.shape),
            'date_pairs': date_pairs,
            'input_window': input_window,
            'forecast_horizon': forecast_horizon,
            'synthetic': True
        }
        
        metadata_file = processed_dir / f"{split_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved: {data_file}")
    
    # Compute normalization stats from training data
    print("\nComputing normalization statistics...")
    train_file = processed_dir / "train_data.npz"
    data = np.load(train_file)
    train_inputs = data['inputs']
    
    ocean_mask = mask == 1
    ocean_data = train_inputs[:, :, ocean_mask]
    
    stats = {
        'mean': float(np.mean(ocean_data)),
        'std': float(np.std(ocean_data))
    }
    
    stats_file = processed_dir / "normalization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Saved: {stats_file}")
    
    print("\n" + "="*60)
    print("DEMO DATA CREATION COMPLETE")
    print("="*60)
    print("Files created:")
    print(f"  - {processed_dir / 'land_ocean_mask.npy'}")
    print(f"  - {processed_dir / 'train_data.npz'} ({len(data['inputs'])} samples)")
    print(f"  - {processed_dir / 'val_data.npz'}")
    print(f"  - {processed_dir / 'test_data.npz'}")
    print(f"  - {processed_dir / 'normalization_stats.json'}")
    print("\nNext step:")
    print("  python scripts/evaluate_baselines.py")
    print("="*60)

if __name__ == "__main__":
    create_demo_dataset()
