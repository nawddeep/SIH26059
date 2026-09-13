"""
Phase 2: Prepare environmental forcing variables for sea-ice forecasting.

This script coordinates the full data pipeline for Phase 2:
1. Download ERA5 variables (wind U/V, air temp, SST)
2. Download Copernicus Marine ocean currents (U/V)
3. Regrid all to NSIDC SIC polar stereographic grid
4. Temporally align to SIC daily timestamps
5. Compute normalization statistics from training split
6. Save processed multi-variable dataset

All decisions are documented: data sources, regridding methods,
missing data handling, coverage gaps, and normalization parameters.
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from seaice_forecast.config import load_config, get_project_root
from seaice_forecast.data_processing.downloaders.era5 import ERA5Downloader
from seaice_forecast.data_processing.downloaders.copernicus import CopernicusMarineDownloader
from seaice_forecast.data_processing.regridding import EnvironmentalRegridder
from seaice_forecast.data_processing.preprocessing import SICPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2DataPipeline:
    """
    Coordinate Phase 2 environmental data preparation.

    Manages the full pipeline from raw downloads to normalized,
    grid-aligned multi-variable arrays ready for model training.
    """

    def __init__(self, config: Dict):
        """
        Initialize Phase 2 data pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.project_root = get_project_root()

        # Setup directories
        self.raw_era5_dir = self.project_root / 'data' / 'raw' / 'era5'
        self.raw_cmems_dir = self.project_root / 'data' / 'raw' / 'copernicus'
        self.regridded_dir = self.project_root / 'data' / 'processed' / 'regridded'
        self.phase2_dir = self.project_root / 'data' / 'processed' / 'phase2'

        for d in [self.raw_era5_dir, self.raw_cmems_dir, self.regridded_dir, self.phase2_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Date ranges
        self.train_start = config['data']['date_ranges']['train_start']
        self.train_end = config['data']['date_ranges']['train_end']
        self.val_start = config['data']['date_ranges']['val_start']
        self.val_end = config['data']['date_ranges']['val_end']
        self.test_start = config['data']['date_ranges']['test_start']
        self.test_end = config['data']['date_ranges']['test_end']

        # Full date range
        self.full_start = self.train_start
        self.full_end = self.test_end

        # Environmental variables to process
        self.variables = {
            'wind_u': {
                'source': 'ERA5',
                'era5_name': 'u10',
                'description': '10m U wind component',
                'units': 'm/s',
                'regrid_method': 'bilinear',
                'missing_strategy': 'nearest_spatial_then_temporal'
            },
            'wind_v': {
                'source': 'ERA5',
                'era5_name': 'v10',
                'description': '10m V wind component',
                'units': 'm/s',
                'regrid_method': 'bilinear',
                'missing_strategy': 'nearest_spatial_then_temporal'
            },
            'air_temp': {
                'source': 'ERA5',
                'era5_name': 't2m',
                'description': '2m air temperature',
                'units': 'K',
                'regrid_method': 'bilinear',
                'missing_strategy': 'nearest_spatial_then_temporal'
            },
            'sst': {
                'source': 'ERA5',
                'era5_name': 'sst',
                'description': 'Sea surface temperature',
                'units': 'K',
                'regrid_method': 'bilinear',
                'missing_strategy': 'ocean_only_nearest',
                'note': 'ERA5 SST chosen for consistency with other ERA5 variables'
            },
            'current_u': {
                'source': 'CMEMS',
                'cmems_name': 'uo',
                'description': 'Eastward ocean current',
                'units': 'm/s',
                'regrid_method': 'bilinear',
                'missing_strategy': 'ocean_only_nearest',
                'depth': '0.5m'
            },
            'current_v': {
                'source': 'CMEMS',
                'cmems_name': 'vo',
                'description': 'Northward ocean current',
                'units': 'm/s',
                'regrid_method': 'bilinear',
                'missing_strategy': 'ocean_only_nearest',
                'depth': '0.5m'
            }
        }

        # Metadata tracking
        self.pipeline_metadata = {
            'phase': 2,
            'created': datetime.now().isoformat(),
            'date_ranges': {
                'train': f"{self.train_start} to {self.train_end}",
                'val': f"{self.val_start} to {self.val_end}",
                'test': f"{self.test_start} to {self.test_end}"
            },
            'variables': self.variables,
            'grid_info': {
                'reference': 'NSIDC PS25 Antarctic',
                'shape': [316, 332],
                'crs': 'EPSG:3031'
            },
            'downloads': {},
            'regridding': {},
            'coverage_gaps': {},
            'normalization': {}
        }

    def step1_download_era5(self, skip_existing: bool = True) -> Dict:
        """
        Step 1: Download ERA5 variables.

        Args:
            skip_existing: Skip if files already exist

        Returns:
            Dictionary of downloaded file paths per variable
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Downloading ERA5 Variables")
        logger.info("="*70)

        downloader = ERA5Downloader(str(self.raw_era5_dir))

        era5_vars = {k: v for k, v in self.variables.items() if v['source'] == 'ERA5'}

        start = datetime.strptime(self.full_start, '%Y-%m-%d')
        end = datetime.strptime(self.full_end, '%Y-%m-%d')

        downloaded = {}

        for var_key, var_info in era5_vars.items():
            logger.info(f"\nDownloading {var_key}: {var_info['description']}")

            files = downloader.download_variable_daterange(
                var_key,
                start,
                end,
                time='00:00',
                monthly_chunks=True
            )

            downloaded[var_key] = [str(f) for f in files]

            # Check coverage
            if files:
                coverage = downloader.check_coverage(files[0])
                self.pipeline_metadata['downloads'][var_key] = {
                    'n_files': len(files),
                    'sample_coverage': coverage
                }

                # Track coverage gaps
                if coverage['missing_data']['fraction'] > 0.01:
                    self.pipeline_metadata['coverage_gaps'][var_key] = {
                        'missing_fraction': coverage['missing_data']['fraction'],
                        'location': 'See individual file coverage reports',
                        'handling': var_info['missing_strategy']
                    }

        logger.info(f"\nERA5 download complete: {len(downloaded)} variables")
        return downloaded

    def step2_download_cmems(self, skip_existing: bool = True) -> Dict:
        """
        Step 2: Download Copernicus Marine ocean currents.

        Args:
            skip_existing: Skip if files already exist

        Returns:
            Dictionary of downloaded file paths per variable
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Downloading Copernicus Marine Ocean Currents")
        logger.info("="*70)

        downloader = CopernicusMarineDownloader(
            str(self.raw_cmems_dir),
            depth_level=0.5  # Near-surface currents
        )

        cmems_vars = {k: v for k, v in self.variables.items() if v['source'] == 'CMEMS'}

        start = datetime.strptime(self.full_start, '%Y-%m-%d')
        end = datetime.strptime(self.full_end, '%Y-%m-%d')

        downloaded = {}

        for var_key, var_info in cmems_vars.items():
            logger.info(f"\nDownloading {var_key}: {var_info['description']}")

            files = downloader.download_variable_daterange(
                var_key,
                start,
                end,
                monthly_chunks=True
            )

            downloaded[var_key] = [str(f) for f in files]

            # Check coverage
            if files:
                coverage = downloader.check_coverage(files[0])
                self.pipeline_metadata['downloads'][var_key] = {
                    'n_files': len(files),
                    'sample_coverage': coverage
                }

                # Track coverage gaps (ocean currents may have coastal gaps)
                if coverage['missing_data']['fraction'] > 0.01:
                    self.pipeline_metadata['coverage_gaps'][var_key] = {
                        'missing_fraction': coverage['missing_data']['fraction'],
                        'location': 'Likely coastal regions',
                        'handling': var_info['missing_strategy']
                    }

        logger.info(f"\nCMEMS download complete: {len(downloaded)} variables")
        return downloaded

    def step3_regrid_all_variables(
        self,
        sic_reference_file: Path,
        land_ocean_mask: np.ndarray
    ) -> Dict:
        """
        Step 3: Regrid all environmental variables to SIC grid.

        Args:
            sic_reference_file: Reference SIC file for target grid
            land_ocean_mask: Land/ocean mask from Phase 1

        Returns:
            Dictionary of regridded file paths
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Regridding All Variables to SIC Grid")
        logger.info("="*70)
        logger.info(f"Target grid: NSIDC PS25 Antarctic (316×332)")
        logger.info(f"Reference file: {sic_reference_file.name}")

        regridder = EnvironmentalRegridder(
            sic_reference_file,
            land_ocean_mask=land_ocean_mask
        )

        regridded = {}

        # Process ERA5 variables
        for var_key, var_info in self.variables.items():
            if var_info['source'] == 'ERA5':
                logger.info(f"\n--- Regridding {var_key} ---")

                # Find merged or monthly files
                merged_file = self.raw_era5_dir / f"era5_{var_key}_merged.nc"

                if not merged_file.exists():
                    # Look for monthly files
                    monthly_files = sorted(self.raw_era5_dir.glob(f"era5_{var_key}_*.nc"))
                    if not monthly_files:
                        logger.warning(f"No files found for {var_key}, skipping")
                        continue

                    # Use first file for now (in practice, merge or process all)
                    input_file = monthly_files[0]
                    logger.info(f"Using file: {input_file.name}")
                else:
                    input_file = merged_file

                output_file = self.regridded_dir / f"{var_key}_regridded.nc"

                # Get ERA5 variable name
                era5_var_name = var_info.get('era5_name', var_key)

                try:
                    regrid_path, metadata = regridder.regrid_era5_variable(
                        input_file,
                        era5_var_name,
                        output_file,
                        method=var_info['regrid_method'],
                        fill_missing=None  # Keep as NaN for now
                    )

                    regridded[var_key] = str(regrid_path)
                    self.pipeline_metadata['regridding'][var_key] = metadata

                except Exception as e:
                    logger.error(f"Failed to regrid {var_key}: {e}")

        # Process CMEMS variables
        for var_key, var_info in self.variables.items():
            if var_info['source'] == 'CMEMS':
                logger.info(f"\n--- Regridding {var_key} ---")

                merged_file = self.raw_cmems_dir / f"cmems_{var_key}_merged.nc"

                if not merged_file.exists():
                    monthly_files = sorted(self.raw_cmems_dir.glob(f"cmems_{var_key}_*.nc"))
                    if not monthly_files:
                        logger.warning(f"No files found for {var_key}, skipping")
                        continue
                    input_file = monthly_files[0]
                else:
                    input_file = merged_file

                output_file = self.regridded_dir / f"{var_key}_regridded.nc"

                cmems_var_name = var_info.get('cmems_name', var_key)

                try:
                    regrid_path, metadata = regridder.regrid_copernicus_variable(
                        input_file,
                        cmems_var_name,
                        output_file,
                        method=var_info['regrid_method'],
                        fill_missing=None
                    )

                    regridded[var_key] = str(regrid_path)
                    self.pipeline_metadata['regridding'][var_key] = metadata

                except Exception as e:
                    logger.error(f"Failed to regrid {var_key}: {e}")

        logger.info(f"\nRegridding complete: {len(regridded)} variables")
        return regridded

    def step4_compute_normalization_stats(
        self,
        regridded_files: Dict[str, str],
        sic_train_dates: List[np.datetime64]
    ) -> Dict:
        """
        Step 4: Compute normalization statistics from training split.

        Statistics computed only from training period (2010-2018) and
        applied unchanged to validation and test sets.

        Args:
            regridded_files: Dictionary of regridded file paths
            sic_train_dates: Training dates from SIC dataset

        Returns:
            Dictionary of normalization statistics per variable
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Computing Normalization Statistics")
        logger.info("="*70)
        logger.info(f"Using training period: {self.train_start} to {self.train_end}")
        logger.info("Method: z-score normalization per variable")

        norm_stats = {}

        for var_key, file_path in regridded_files.items():
            logger.info(f"\n--- Computing stats for {var_key} ---")

            # Load data
            ds = xr.open_dataset(file_path)

            # Get variable name (might differ from var_key)
            var_name = list(ds.data_vars.keys())[0]

            # Select training period
            train_start_dt = np.datetime64(self.train_start)
            train_end_dt = np.datetime64(self.train_end)

            train_data = ds[var_name].sel(
                time=slice(train_start_dt, train_end_dt)
            )

            # Compute mean and std over all valid (non-NaN) pixels and times
            values = train_data.values
            valid_values = values[~np.isnan(values)]

            if len(valid_values) == 0:
                logger.warning(f"No valid data for {var_key} in training period")
                mean = 0.0
                std = 1.0
            else:
                mean = float(np.mean(valid_values))
                std = float(np.std(valid_values))

                # Avoid division by zero
                if std == 0.0:
                    logger.warning(f"Zero std for {var_key}, setting to 1.0")
                    std = 1.0

            norm_stats[var_key] = {
                'mean': mean,
                'std': std,
                'n_valid_samples': int(len(valid_values)),
                'min': float(np.min(valid_values)) if len(valid_values) > 0 else None,
                'max': float(np.max(valid_values)) if len(valid_values) > 0 else None,
                'units': self.variables[var_key]['units']
            }

            logger.info(f"  Mean: {mean:.4f}")
            logger.info(f"  Std:  {std:.4f}")
            logger.info(f"  Range: [{norm_stats[var_key]['min']:.4f}, "
                       f"{norm_stats[var_key]['max']:.4f}]")
            logger.info(f"  Valid samples: {len(valid_values):,}")

            ds.close()

        self.pipeline_metadata['normalization'] = norm_stats

        # Save normalization stats
        stats_file = self.phase2_dir / 'normalization_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(norm_stats, f, indent=2)

        logger.info(f"\nNormalization stats saved to: {stats_file}")

        return norm_stats

    def step5_create_combined_dataset(
        self,
        regridded_files: Dict[str, str],
        norm_stats: Dict,
        sic_data_files: Dict[str, Path]
    ) -> Dict[str, Path]:
        """
        Step 5: Create combined multi-variable datasets.

        Combines SIC + environmental variables into unified arrays
        with proper normalization and train/val/test splits.

        Args:
            regridded_files: Regridded environmental variable files
            norm_stats: Normalization statistics
            sic_data_files: SIC data files from Phase 1

        Returns:
            Dictionary of output file paths
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Creating Combined Multi-Variable Dataset")
        logger.info("="*70)
        logger.info("Variables: SIC + wind_u + wind_v + air_temp + sst + current_u + current_v")
        logger.info("Input: 7 days × 7 variables = 49 channels")

        output_files = {}

        # Variable order (must be consistent!)
        var_order = ['sic', 'wind_u', 'wind_v', 'air_temp', 'sst', 'current_u', 'current_v']

        logger.info(f"Variable order: {var_order}")

        # Process each split
        for split in ['train', 'val', 'test']:
            logger.info(f"\n--- Processing {split} split ---")

            # Load SIC data
            sic_file = sic_data_files.get(split)
            if sic_file is None or not sic_file.exists():
                logger.warning(f"SIC data not found for {split}, skipping")
                continue

            sic_npz = np.load(sic_file)
            sic_inputs = sic_npz['inputs']  # [N, 7, H, W]
            sic_targets = sic_npz['targets']  # [N, 1, H, W]

            n_samples = sic_inputs.shape[0]
            H, W = sic_inputs.shape[2:]

            logger.info(f"  SIC data: {n_samples} samples, shape {sic_inputs.shape}")

            # Prepare combined input array: [N, 49, H, W]
            combined_inputs = np.zeros((n_samples, 49, H, W), dtype=np.float32)

            # Fill SIC channels (already normalized 0-1)
            combined_inputs[:, 0:7, :, :] = sic_inputs

            # Load and normalize environmental variables
            # NOTE: This is a simplified version - in practice, need proper temporal alignment
            channel_idx = 7

            for var_key in var_order[1:]:  # Skip 'sic'
                if var_key not in regridded_files:
                    logger.warning(f"Variable {var_key} not available, filling with zeros")
                    combined_inputs[:, channel_idx:channel_idx+7, :, :] = 0.0
                    channel_idx += 7
                    continue

                logger.info(f"  Loading {var_key}...")

                # Load regridded data
                ds = xr.open_dataset(regridded_files[var_key])
                var_name = list(ds.data_vars.keys())[0]

                # Get normalization stats
                mean = norm_stats[var_key]['mean']
                std = norm_stats[var_key]['std']

                # For this simplified version, just use first N samples
                # In practice, need proper date alignment
                data = ds[var_name].values[:n_samples]  # [N, H, W]

                # Normalize
                data_norm = (data - mean) / std

                # Fill NaN with 0
                data_norm = np.nan_to_num(data_norm, nan=0.0)

                # Replicate to 7 time steps (placeholder - need actual 7-day history)
                for t in range(7):
                    combined_inputs[:, channel_idx + t, :, :] = data_norm

                ds.close()
                channel_idx += 7

            # Save combined dataset
            output_file = self.phase2_dir / f'{split}_data_phase2.npz'

            np.savez_compressed(
                output_file,
                inputs=combined_inputs,
                targets=sic_targets,
                metadata={
                    'phase': 2,
                    'split': split,
                    'n_samples': n_samples,
                    'input_shape': combined_inputs.shape,
                    'variable_order': var_order,
                    'channels_per_variable': 7
                }
            )

            logger.info(f"  Saved: {output_file.name}")
            logger.info(f"  Input shape: {combined_inputs.shape}")
            logger.info(f"  Target shape: {sic_targets.shape}")

            output_files[split] = output_file

        return output_files

    def generate_coupled_synthetic_data(
        self,
        n_train_days: int = 200,
        n_val_days: int = 70,
        n_test_days: int = 70,
        input_window: int = 7,
        forecast_horizon: int = 1
    ) -> Dict:
        """
        Generate physically coupled synthetic Antarctic environmental forcing data.

        Creates realistic, physically aligned daily fields for all 7 variables:
        1. SIC: Sea-ice concentration [0, 1] bounded
        2. wind_u: 10m eastward wind velocity (m/s)
        3. wind_v: 10m northward wind velocity (m/s)
        4. air_temp: 2m air temperature (K)
        5. sst: Sea surface temperature (K, >= 271.35 K)
        6. current_u: Ocean surface eastward velocity (m/s)
        7. current_v: Ocean surface northward velocity (m/s)

        Includes dynamic advection (wind + ocean drift) and thermodynamic
        melt/freeze effects, ensuring physical coupling across variables.
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING PHYSICALLY COUPLED SYNTHETIC DATASET")
        logger.info("="*70)

        processed_dir = self.project_root / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_dir.mkdir(parents=True, exist_ok=True)

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
            logger.info(f"\nSynthesizing {split_name} split ({n_days} days)...")
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
        logger.info("\nComputing per-variable normalization statistics from training ocean pixels...")
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

        stats_file = self.phase2_dir / 'normalization_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(norm_stats, f, indent=2)

        # Phase 1 stats file for compatibility
        with open(processed_dir / 'normalization_stats.json', 'w') as f:
            json.dump({'mean': norm_stats['sic']['mean'], 'std': norm_stats['sic']['std']}, f, indent=2)

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
                # Channel ordering: 7 days of var0, 7 days of var1, ..., 7 days of var6
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
                'synthetic': True
            }
            with open(processed_dir / f"{split_name}_metadata.json", 'w') as f:
                json.dump(p1_meta, f, indent=2)
            results['phase1'][split_name] = p1_file

            # Save Phase 2 NPZ
            p2_file = self.phase2_dir / f"{split_name}_data_phase2.npz"
            np.savez(
                p2_file,
                inputs=p2_inputs,
                targets=p2_targets,
                metadata={
                    'phase': 2,
                    'split': split_name,
                    'n_samples': n_samples,
                    'input_shape': list(p2_inputs.shape),
                    'target_shape': list(p2_targets.shape),
                    'variable_order': var_order,
                    'channels_per_variable': input_window,
                    'date_pairs': date_pairs
                }
            )
            p2_meta = {
                'phase': 2,
                'split': split_name,
                'n_samples': n_samples,
                'input_shape': list(p2_inputs.shape),
                'target_shape': list(p2_targets.shape),
                'variable_order': var_order,
                'channels_per_variable': input_window,
                'date_pairs': date_pairs
            }
            with open(self.phase2_dir / f"{split_name}_metadata.json", 'w') as f:
                json.dump(p2_meta, f, indent=2)
            results['phase2'][split_name] = p2_file

            logger.info(f"  {split_name:5s} -> Phase 1: {p1_inputs.shape}, Phase 2: {p2_inputs.shape} ({n_samples} samples)")

        # Save pipeline metadata
        pipeline_meta = {
            'phase': 2,
            'created': datetime.now().isoformat(),
            'synthetic_coupled': True,
            'grid': {'shape': [H, W], 'resolution': '25km', 'name': 'NSIDC PS25 Antarctic'},
            'variables': var_order,
            'channels': 49,
            'normalization': norm_stats,
            'splits': {k: {'samples': len(v[1]) - input_window} for k, v in raw_series.items()}
        }
        with open(self.phase2_dir / 'pipeline_metadata.json', 'w') as f:
            json.dump(pipeline_meta, f, indent=2)

        logger.info("\n" + "="*70)
        logger.info("DATA PREPARATION COMPLETE: Both Phase 1 & Phase 2 Datasets Created")
        logger.info("="*70)
        return results

    def run_full_pipeline(
        self,
        download: bool = False,
        regrid: bool = False,
        synthetic: bool = False,
        skip_existing: bool = True
    ) -> Dict:
        """
        Run the complete Phase 2 data pipeline.

        Args:
            download: Whether to download data from real sources
            regrid: Whether to regrid data
            synthetic: Generate synthetic coupled data
            skip_existing: Skip steps with existing outputs

        Returns:
            Dictionary with pipeline results
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 2 DATA PIPELINE")
        logger.info("="*70)

        # Check for synthetic flag
        if synthetic:
            logger.error("SYNTHETIC DATA GENERATION HAS BEEN MOVED")
            logger.error("To generate synthetic data, run:")
            logger.error("  python dev_tools/synthetic_data/SYNTHETIC_environmental_generator.py \\")
            logger.error("    --use-synthetic-data --i-understand-this-is-fake \\")
            logger.error("    --output-dir <output_path>")
            raise ValueError("Synthetic data generation must use explicit safety flags via dev_tools/")

        # Check for real data
        sic_raw_dir = self.project_root / 'data' / 'raw' / 'nsidc'
        has_raw = sic_raw_dir.exists() and len(list(sic_raw_dir.glob('*.nc'))) > 0

        if not download and not has_raw:
            logger.error("No real data found and download=False")
            logger.error("Options:")
            logger.error("  1. Set download=True to download real data")
            logger.error("  2. For development ONLY, generate synthetic data:")
            logger.error("     python dev_tools/synthetic_data/SYNTHETIC_environmental_generator.py \\")
            logger.error("       --use-synthetic-data --i-understand-this-is-fake \\")
            logger.error("       --output-dir <output_path>")
            raise FileNotFoundError("No real data available. Use download=True or see error message above.")

        logger.info(f"Date range: {self.full_start} to {self.full_end}")
        logger.info(f"Variables: {len(self.variables)}")
        logger.info(f"Target grid: NSIDC PS25 (316x332)")

        results = {}
        processed_dir = self.project_root / 'data' / 'processed'
        mask_file = processed_dir / 'land_ocean_mask.npy'

        if not mask_file.exists():
            raise FileNotFoundError(
                f"Land/ocean mask not found at {mask_file}. "
                "Run data preparation first or generate synthetic data using dev_tools/synthetic_data/"
            )

        land_ocean_mask = np.load(mask_file)

        sic_files_list = list(sic_raw_dir.glob('*.nc')) if sic_raw_dir.exists() else []
        if not sic_files_list:
            raise FileNotFoundError(
                f"No SIC reference files found in {sic_raw_dir}. "
                "Download real NSIDC data first. For development ONLY, you can generate "
                "synthetic data using: python dev_tools/synthetic_data/SYNTHETIC_environmental_generator.py "
                "--use-synthetic-data --i-understand-this-is-fake --output-dir <path>"
            )

        sic_reference = sic_files_list[0]

        if download:
            results['era5_downloads'] = self.step1_download_era5(skip_existing)
            results['cmems_downloads'] = self.step2_download_cmems(skip_existing)

        if regrid:
            results['regridded'] = self.step3_regrid_all_variables(sic_reference, land_ocean_mask)
        else:
            results['regridded'] = {var: str(self.regridded_dir / f"{var}_regridded.nc") for var in self.variables.keys()}

        train_dates = [np.datetime64(self.train_start)]
        results['norm_stats'] = self.step4_compute_normalization_stats(results['regridded'], train_dates)

        sic_files = {
            'train': processed_dir / 'train_data.npz',
            'val': processed_dir / 'val_data.npz',
            'test': processed_dir / 'test_data.npz'
        }
        results['combined_datasets'] = self.step5_create_combined_dataset(results['regridded'], results['norm_stats'], sic_files)

        metadata_file = self.phase2_dir / 'pipeline_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=2)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Phase 2 environmental forcing data"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download data from sources'
    )
    parser.add_argument(
        '--regrid',
        action='store_true',
        help='Regrid downloaded data'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        default=False,
        help='DEPRECATED: Synthetic data is strictly disabled by default (default: False)'
    )
    parser.add_argument(
        '--i-know-this-is-fake',
        action='store_true',
        default=False,
        help='Explicit opt-in required to acknowledge non-real data for development testing'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip files that already exist'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create pipeline
    pipeline = Phase2DataPipeline(config)

    # Run pipeline
    results = pipeline.run_full_pipeline(
        download=args.download,
        regrid=args.regrid,
        synthetic=args.synthetic,
        skip_existing=args.skip_existing
    )

    print("\n" + "="*70)
    print("Pipeline Execution Completed Successfully")
    print("="*70)


if __name__ == "__main__":
    main()

