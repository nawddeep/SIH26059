#!/usr/bin/env python
"""
⚠️ ⚠️ ⚠️  SYNTHETIC ENVIRONMENTAL DATA GENERATOR - NOT FOR PRODUCTION SCIENCE  ⚠️ ⚠️ ⚠️

This generates FAKE coupled environmental forcing data (SIC + wind + temp + SST + currents).
DO NOT use this for any scientific analysis, publication, or operational forecasting.

This tool creates physically plausible but entirely fabricated multi-variable fields.
It exists ONLY to enable Phase 2 development workflows when real ERA5/Copernicus data is unavailable.

REQUIRED FLAGS TO RUN:
  --use-synthetic-data --i-understand-this-is-fake

Without these explicit flags, this script will REFUSE to run.
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "iceberg_forcasting_model" / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def generate_FAKE_coupled_environmental_data(
    output_dir: Path,
    n_train_days: int = 200,
    n_val_days: int = 70,
    n_test_days: int = 70,
    input_window: int = 7,
    forecast_horizon: int = 1
):
    """
    Generate FAKE physically coupled synthetic Antarctic environmental forcing data.

    ⚠️  ALL DATA GENERATED IS FABRICATED - NOT FOR REAL SCIENCE  ⚠️

    Creates realistic-looking, physically aligned daily fields for all 7 variables:
    1. SIC: Sea-ice concentration [0, 1] bounded
    2. wind_u: 10m eastward wind velocity (m/s)
    3. wind_v: 10m northward wind velocity (m/s)
    4. air_temp: 2m air temperature (K)
    5. sst: Sea surface temperature (K, >= 271.35 K)
    6. current_u: Ocean surface eastward velocity (m/s)
    7. current_v: Ocean surface northward velocity (m/s)

    Includes dynamic advection (wind + ocean drift) and thermodynamic
    melt/freeze effects, ensuring physical coupling across variables.

    BUT IT'S ALL FAKE!
    """
    logger.info("\n" + "="*70)
    logger.info("⚠️  GENERATING FAKE COUPLED ENVIRONMENTAL DATASET  ⚠️")
    logger.info("="*70)
    logger.info("This data is FABRICATED for development purposes only.")
    logger.info("DO NOT use for science, publications, or operational forecasts.")
    logger.info("="*70)

    processed_dir = output_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    phase2_dir = output_dir / 'processed' / 'phase2'
    phase2_dir.mkdir(parents=True, exist_ok=True)

    H, W = 316, 332
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist_from_pole = np.sqrt((y - cy)**2 + (x - cx)**2)
    theta = np.arctan2(y - cy, x - cx)

    # Load or create land-ocean mask (1=ocean, 0=land)
    mask_file = processed_dir / 'land_ocean_mask.npy'
    if mask_file.exists():
        mask = np.load(mask_file)
        logger.info(f"Loaded existing land-ocean mask: {mask.shape}")
    else:
        mask = np.ones((H, W), dtype=np.float32)
        # Central Antarctic continent + outer borders
        continent = dist_from_pole < 45
        mask[continent] = 0.0
        mask[:15, :] = 0.0
        mask[-15:, :] = 0.0
        mask[:, :15] = 0.0
        mask[:, -15:] = 0.0
        np.save(mask_file, mask)
        logger.info(f"Created land-ocean mask: {np.sum(mask == 1)} ocean pixels")

    ocean_mask = (mask == 1)

    # Define splits with base start dates
    splits_info = {
        'train': (n_train_days, datetime(2018, 1, 1), 42),
        'val': (n_val_days, datetime(2019, 1, 1), 142),
        'test': (n_test_days, datetime(2021, 1, 1), 242)
    }

    raw_series = {}
    var_order = ['sic', 'wind_u', 'wind_v', 'air_temp', 'sst', 'current_u', 'current_v']

    for split_name, (n_days, base_date, seed) in splits_info.items():
        logger.info(f"\nSynthesizing FAKE {split_name} split ({n_days} days)...")
        np.random.seed(seed)

        # Arrays to store daily fields: [n_days, H, W]
        daily_vars = {v: np.zeros((n_days, H, W), dtype=np.float32) for v in var_order}
        dates_list = []

        # Initial state
        base_edge_radius = 85.0
        prev_sic = np.clip(1.0 - (dist_from_pole / (base_edge_radius + 5.0)) ** 1.5, 0.0, 1.0) * mask

        for d in range(n_days):
            current_date = base_date + timedelta(days=d)
            dates_list.append(current_date.strftime("%Y-%m-%d"))
            doy = current_date.timetuple().tm_yday
            seasonal_phase = 2.0 * np.pi * (doy - 45) / 365.25  # Min ice in Feb, max in Aug/Sep

            # Dynamic edge radius governed by season
            seasonal_radius = base_edge_radius + 28.0 * np.sin(seasonal_phase)

            # 1. Atmospheric Wind: Circumpolar westerlies + travelling cyclonic storm
            v_circ = 9.0 + 3.0 * np.sin(seasonal_phase)
            u_circ = -v_circ * np.sin(theta)
            v_circ_y = v_circ * np.cos(theta)

            # Travelling cyclone
            storm_theta = theta - (0.15 * d) % (2.0 * np.pi)
            storm_factor = np.exp(-((dist_from_pole - 95.0)**2) / 600.0) * np.sin(2.0 * storm_theta)

            u_wind = u_circ + 4.0 * storm_factor + np.random.randn(H, W) * 1.5
            v_wind = v_circ_y - 4.0 * storm_factor + np.random.randn(H, W) * 1.5

            # 2. Ocean Surface Currents: ACC + wind stress coupling
            u_curr = 0.02 * u_wind - 0.18 * np.sin(theta) + np.random.randn(H, W) * 0.04
            v_curr = 0.02 * v_wind + 0.18 * np.cos(theta) + np.random.randn(H, W) * 0.04

            # 3. 2m Air Temperature (K): Polar cold core, Southern Ocean maritime warmth
            t_base = 248.0 + 32.0 * (dist_from_pole / (dist_from_pole.max() + 1e-5))
            t_season = -14.0 * np.sin(seasonal_phase)  # Winter cold, summer mild
            t_advection = -0.4 * (v_wind * np.sin(theta) + u_wind * np.cos(theta))  # Wind from south is cold
            air_temp = t_base + t_season + t_advection + np.random.randn(H, W) * 0.8

            # 4. Sea Surface Temperature (K): Freezing near ice, warmer northward
            sst_freezing = 271.35  # -1.8 C
            open_ocean_dist = np.maximum(0.0, dist_from_pole - seasonal_radius)
            sst = sst_freezing + 7.5 * (1.0 - np.exp(-open_ocean_dist / 35.0)) + 1.2 * np.sin(seasonal_phase)
            sst = np.maximum(sst_freezing, sst + np.random.randn(H, W) * 0.15)

            # 5. Sea Ice Concentration (SIC): Advection + Thermodynamic response
            # Advection shift
            drift_x = (0.015 * u_wind + 0.45 * u_curr) * 1.2
            drift_y = (0.015 * v_wind + 0.45 * v_curr) * 1.2

            mean_dx = float(np.mean(drift_x[ocean_mask]))
            mean_dy = float(np.mean(drift_y[ocean_mask]))

            # Shift SIC by integer pixel approximation + diffusion
            shift_y = int(np.clip(round(mean_dy), -2, 2))
            shift_x = int(np.clip(round(mean_dx), -2, 2))

            advected_sic = np.roll(np.roll(prev_sic, shift_y, axis=0), shift_x, axis=1)

            # Thermodynamics: freezing where air_temp < 271.35 and sst near freezing
            freeze_melt = np.where(air_temp < 271.35, 0.015 * (271.35 - air_temp) / 10.0, -0.025 * (air_temp - 271.35) / 5.0)
            freeze_melt = np.where(sst > 272.5, -0.04 * (sst - 271.35), freeze_melt)

            # Base radial envelope
            edge_mask = np.clip(1.0 - (dist_from_pole / (seasonal_radius + 5.0))**2.0, 0.0, 1.0)
            sic = np.clip(0.85 * advected_sic + 0.15 * edge_mask + freeze_melt + np.random.randn(H, W) * 0.02, 0.0, 1.0)
            sic = sic * mask

            # Save daily values
            daily_vars['sic'][d] = sic
            daily_vars['wind_u'][d] = u_wind * mask
            daily_vars['wind_v'][d] = v_wind * mask
            daily_vars['air_temp'][d] = air_temp * mask
            daily_vars['sst'][d] = sst * mask
            daily_vars['current_u'][d] = u_curr * mask
            daily_vars['current_v'][d] = v_curr * mask

            prev_sic = sic

        raw_series[split_name] = (daily_vars, dates_list)

    # Step: Compute normalization statistics strictly from training ocean pixels
    logger.info("\nComputing per-variable normalization statistics from FAKE training ocean pixels...")
    train_daily = raw_series['train'][0]
    norm_stats = {}

    for v in var_order:
        if v == 'sic':
            norm_stats['sic'] = {'mean': 0.0, 'std': 1.0, 'scaling': 'bounded [0, 1]'}
        else:
            v_data = train_daily[v][:, ocean_mask]
            norm_stats[v] = {
                'mean': float(np.mean(v_data)),
                'std': float(np.std(v_data) if np.std(v_data) > 1e-6 else 1.0)
            }
        logger.info(f"  {v:10s} -> mean: {norm_stats[v]['mean']:.4f}, std: {norm_stats[v]['std']:.4f}")

    # Add warning to stats
    for v in norm_stats:
        norm_stats[v]['WARNING'] = 'COMPUTED FROM SYNTHETIC DATA - NOT REAL'
        norm_stats[v]['synthetic'] = True

    stats_file = phase2_dir / 'normalization_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # Phase 1 stats file for compatibility
    with open(processed_dir / 'normalization_stats.json', 'w') as f:
        json.dump({
            'mean': norm_stats['sic']['mean'],
            'std': norm_stats['sic']['std'],
            'WARNING': 'SYNTHETIC DATA',
            'synthetic': True
        }, f, indent=2)

    # Step: Construct sliding windows and save both Phase 1 and Phase 2 datasets
    logger.info("\nConstructing sliding windows (7 days history -> day 8 target)...")
    results = {'phase1': {}, 'phase2': {}}

    for split_name, (daily_vars, dates_list) in raw_series.items():
        n_days = len(dates_list)
        n_samples = n_days - input_window - forecast_horizon + 1

        # Normalized arrays for Phase 2
        norm_daily = {}
        for v in var_order:
            if v == 'sic':
                norm_daily[v] = daily_vars[v].copy()
            else:
                norm_daily[v] = (daily_vars[v] - norm_stats[v]['mean']) / norm_stats[v]['std']
                norm_daily[v] = norm_daily[v] * mask  # Re-mask land to 0

        # Construct Phase 1 arrays: inputs [n_samples, 7, H, W], targets [n_samples, 1, H, W]
        p1_inputs = np.zeros((n_samples, input_window, H, W), dtype=np.float32)
        p1_targets = np.zeros((n_samples, forecast_horizon, H, W), dtype=np.float32)

        # Construct Phase 2 arrays: inputs [n_samples, 49, H, W], targets [n_samples, 1, H, W]
        p2_inputs = np.zeros((n_samples, input_window * len(var_order), H, W), dtype=np.float32)
        p2_targets = np.zeros((n_samples, forecast_horizon, H, W), dtype=np.float32)

        date_pairs = []
        for i in range(n_samples):
            # Phase 1 (SIC only)
            p1_inputs[i] = daily_vars['sic'][i : i + input_window]
            p1_targets[i] = daily_vars['sic'][i + input_window : i + input_window + forecast_horizon]

            # Phase 2 (7 variables x 7 days = 49 channels)
            for v_idx, v_name in enumerate(var_order):
                ch_start = v_idx * input_window
                ch_end = ch_start + input_window
                p2_inputs[i, ch_start:ch_end] = norm_daily[v_name][i : i + input_window]

            p2_targets[i] = p1_targets[i]

            input_end = dates_list[i + input_window - 1]
            target_date = dates_list[i + input_window + forecast_horizon - 1]
            date_pairs.append((input_end, target_date))

        # Save Phase 1 NPZ
        p1_file = processed_dir / f"{split_name}_data.npz"
        np.savez(p1_file, inputs=p1_inputs, targets=p1_targets)
        p1_meta = {
            'split': split_name,
            'n_samples': n_samples,
            'input_shape': list(p1_inputs.shape),
            'target_shape': list(p1_targets.shape),
            'date_pairs': date_pairs,
            'input_window': input_window,
            'forecast_horizon': forecast_horizon,
            'WARNING': 'THIS IS SYNTHETIC (FAKE) DATA - NOT FOR REAL SCIENCE',
            'synthetic': True,
            'data_source': 'FABRICATED_DEVELOPMENT_ONLY'
        }
        with open(processed_dir / f"{split_name}_metadata.json", 'w') as f:
            json.dump(p1_meta, f, indent=2)
        results['phase1'][split_name] = str(p1_file)

        # Save Phase 2 NPZ
        p2_file = phase2_dir / f"{split_name}_data_phase2.npz"
        np.savez(
            p2_file,
            inputs=p2_inputs,
            targets=p2_targets
        )
        p2_meta = {
            'phase': 2,
            'split': split_name,
            'n_samples': n_samples,
            'input_shape': list(p2_inputs.shape),
            'target_shape': list(p2_targets.shape),
            'variable_order': var_order,
            'channels_per_variable': input_window,
            'date_pairs': date_pairs,
            'WARNING': 'THIS IS SYNTHETIC (FAKE) DATA - NOT FOR REAL SCIENCE',
            'synthetic': True,
            'data_source': 'FABRICATED_DEVELOPMENT_ONLY'
        }
        with open(phase2_dir / f"{split_name}_metadata.json", 'w') as f:
            json.dump(p2_meta, f, indent=2)
        results['phase2'][split_name] = str(p2_file)

        logger.info(f"  {split_name:5s} -> Phase 1: {p1_inputs.shape}, Phase 2: {p2_inputs.shape} ({n_samples} samples)")

    # Save pipeline metadata
    pipeline_meta = {
        'phase': 2,
        'created': datetime.now().isoformat(),
        'WARNING': 'ALL DATA IS SYNTHETIC (FAKE) - NOT FOR REAL SCIENCE',
        'synthetic_coupled': True,
        'data_source': 'FABRICATED_DEVELOPMENT_ONLY',
        'grid': {'shape': [H, W], 'resolution': '25km', 'name': 'NSIDC PS25 Antarctic'},
        'variables': var_order,
        'channels': 49,
        'normalization': norm_stats,
        'splits': {k: {'samples': len(v[1]) - input_window} for k, v in raw_series.items()}
    }
    with open(phase2_dir / 'pipeline_metadata.json', 'w') as f:
        json.dump(pipeline_meta, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("⚠️  FAKE DATA PREPARATION COMPLETE  ⚠️")
    logger.info("="*70)
    logger.info("Created Both Phase 1 & Phase 2 Datasets (ALL SYNTHETIC)")
    logger.info("\n⚠️  REMINDER: This is FAKE data for development only! ⚠️")
    logger.info("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="⚠️  SYNTHETIC ENVIRONMENTAL GENERATOR - Generate FAKE multi-variable environmental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️ ⚠️ ⚠️  CRITICAL WARNING  ⚠️ ⚠️ ⚠️

This script generates FABRICATED multi-variable environmental data that looks
realistic but is entirely fake. It exists ONLY to enable Phase 2 development
and testing when real ERA5/Copernicus data is unavailable.

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
        required=True,
        help='Output directory for synthetic data (will create data/ subdirectory)'
    )

    parser.add_argument(
        '--n-train-days',
        type=int,
        default=200,
        help='Number of training days to generate (default: 200)'
    )

    parser.add_argument(
        '--n-val-days',
        type=int,
        default=70,
        help='Number of validation days to generate (default: 70)'
    )

    parser.add_argument(
        '--n-test-days',
        type=int,
        default=70,
        help='Number of test days to generate (default: 70)'
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
        print("  python SYNTHETIC_environmental_generator.py \\")
        print("    --use-synthetic-data \\")
        print("    --i-understand-this-is-fake \\")
        print("    --output-dir /path/to/output")
        print("="*70)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    generate_FAKE_coupled_environmental_data(
        output_dir=output_dir,
        n_train_days=args.n_train_days,
        n_val_days=args.n_val_days,
        n_test_days=args.n_test_days
    )


if __name__ == "__main__":
    main()
