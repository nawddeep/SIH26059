#!/usr/bin/env python
"""
⚠️ ⚠️ ⚠️  SYNTHETIC DATA GENERATOR - NOT FOR PRODUCTION SCIENCE  ⚠️ ⚠️ ⚠️

This generates FAKE sea-ice concentration data for development and testing ONLY.
DO NOT use this for any scientific analysis, publication, or operational forecasting.

This tool creates physically plausible but entirely fabricated SIC fields.
It exists ONLY to enable development workflows when real NSIDC data is unavailable.

REQUIRED FLAGS TO RUN:
  --use-synthetic-data --i-understand-this-is-fake

Without these explicit flags, this script will refuse to run.
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "iceberg_forcasting_model" / "src"))

from seaice_forecast.config import load_config, resolve_paths


def generate_FAKE_sic_for_dev_only(H=316, W=332, seed=42):
    """
    Generate FAKE synthetic SIC field for development testing ONLY.

    ⚠️  THIS IS NOT REAL DATA ⚠️
    """
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


def create_FAKE_land_ocean_mask(H=316, W=332):
    """Create simple FAKE land-ocean mask for development."""
    mask = np.ones((H, W))

    # Make outer edge "land" (Antarctica continent)
    mask[:20, :] = 0
    mask[-20:, :] = 0
    mask[:, :20] = 0
    mask[:, -20:] = 0

    return mask


def create_SYNTHETIC_demo_dataset(output_dir=None):
    """
    Create complete FAKE demo dataset for development.

    ⚠️  ALL DATA GENERATED HERE IS SYNTHETIC AND NOT REAL  ⚠️
    """
    print("\n" + "="*70)
    print("⚠️  CREATING SYNTHETIC (FAKE) DEMO DATA ⚠️")
    print("="*70)
    print("This data is FABRICATED for development purposes only.")
    print("DO NOT use for science, publications, or operational forecasts.")
    print("="*70 + "\n")

    # Load config
    config = load_config()
    config = resolve_paths(config)

    if output_dir is None:
        processed_dir = Path(config['data']['paths']['processed'])
    else:
        processed_dir = Path(output_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    H, W = 316, 332
    input_window = 7
    forecast_horizon = 1

    # Create land-ocean mask
    print("Creating FAKE land-ocean mask...")
    mask = create_FAKE_land_ocean_mask(H, W)
    np.save(processed_dir / "land_ocean_mask.npy", mask)
    print(f"  Mask: {mask.shape}, ocean pixels: {np.sum(mask)}")

    # Generate datasets
    splits = {
        'train': 365,  # 1 year
        'val': 120,    # ~4 months
        'test': 120    # ~4 months
    }

    for split_name, n_days in splits.items():
        print(f"\nCreating FAKE {split_name} split ({n_days} days)...")

        # Generate time series
        sic_sequence = []
        base_date = datetime(2020, 1, 1)

        for day in range(n_days):
            # Add temporal variation
            seed = day + (100 if split_name == 'val' else 200 if split_name == 'test' else 0)
            sic = generate_FAKE_sic_for_dev_only(H, W, seed=seed)

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

        # Save metadata - CLEARLY MARK AS SYNTHETIC
        metadata = {
            'split': split_name,
            'n_samples': len(inputs),
            'input_shape': list(inputs.shape),
            'target_shape': list(targets.shape),
            'date_pairs': date_pairs,
            'input_window': input_window,
            'forecast_horizon': forecast_horizon,
            'WARNING': 'THIS IS SYNTHETIC (FAKE) DATA - NOT FOR REAL SCIENCE',
            'synthetic': True,
            'data_source': 'FABRICATED_DEVELOPMENT_ONLY'
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
        'std': float(np.std(ocean_data)),
        'WARNING': 'COMPUTED FROM SYNTHETIC DATA - NOT REAL',
        'synthetic': True
    }

    stats_file = processed_dir / "normalization_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Saved: {stats_file}")

    print("\n" + "="*70)
    print("⚠️  SYNTHETIC (FAKE) DATA CREATION COMPLETE ⚠️")
    print("="*70)
    print("Files created:")
    print(f"  - {processed_dir / 'land_ocean_mask.npy'}")
    print(f"  - {processed_dir / 'train_data.npz'} ({len(data['inputs'])} samples)")
    print(f"  - {processed_dir / 'val_data.npz'}")
    print(f"  - {processed_dir / 'test_data.npz'}")
    print(f"  - {processed_dir / 'normalization_stats.json'}")
    print("\n⚠️  REMINDER: This is FAKE data for development only! ⚠️")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="⚠️  SYNTHETIC DATA GENERATOR - Generate FAKE sea-ice data for development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️ ⚠️ ⚠️  CRITICAL WARNING  ⚠️ ⚠️ ⚠️

This script generates FABRICATED data that looks realistic but is entirely fake.
It exists ONLY to enable development and testing when real data is unavailable.

NEVER use synthetic data for:
  - Scientific publications
  - Operational forecasting
  - Policy decisions
  - Any analysis claiming to represent real-world conditions

REQUIRED FLAGS:
  You MUST provide both flags to run this script:
    --use-synthetic-data --i-understand-this-is-fake

  This requirement ensures you cannot accidentally generate fake data.
"""
    )

    parser.add_argument(
        '--use-synthetic-data',
        action='store_true',
        required=True,
        help='Explicit acknowledgment that you want to generate FAKE data'
    )

    parser.add_argument(
        '--i-understand-this-is-fake',
        action='store_true',
        required=True,
        help='Explicit acknowledgment that you understand this data is NOT REAL'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for synthetic data (default: uses config)'
    )

    args = parser.parse_args()

    # Double-check both flags are present
    if not (args.use_synthetic_data and args.i_understand_this_is_fake):
        print("\n" + "="*70)
        print("ERROR: Missing required safety flags")
        print("="*70)
        print("This script generates FAKE data and requires explicit consent.")
        print("\nYou MUST provide BOTH flags:")
        print("  --use-synthetic-data")
        print("  --i-understand-this-is-fake")
        print("\nExample:")
        print("  python SYNTHETIC_seaice_generator.py \\")
        print("    --use-synthetic-data \\")
        print("    --i-understand-this-is-fake")
        print("="*70)
        sys.exit(1)

    create_SYNTHETIC_demo_dataset(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
