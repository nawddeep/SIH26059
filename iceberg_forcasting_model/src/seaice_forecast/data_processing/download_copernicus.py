"""
Copernicus Marine Service data download module.

Downloads ocean current U/V components for Antarctic sea-ice forecasting.

Data source: Copernicus Marine Environment Monitoring Service (CMEMS)
Product: GLOBAL_MULTIYEAR_PHY_001_030 (GLORYS12V1 reanalysis)
Variables: Eastward (U) and Northward (V) ocean current velocity

Requires: copernicusmarine library and valid CMEMS credentials
Install: pip install copernicusmarine
Setup: copernicusmarine login
"""

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CopernicusMarineDownloader:
    """
    Download ocean current data from Copernicus Marine Service.

    Uses the GLORYS12V1 global ocean reanalysis at 1/12° resolution.
    """

    def __init__(
        self,
        output_dir: str,
        antarctic_bounds: Tuple[float, float, float, float] = (-90, 0, -50, 360),
        depth_level: float = 0.5  # Surface currents (0.5m depth)
    ):
        """
        Initialize Copernicus Marine downloader.

        Args:
            output_dir: Directory to store downloaded files
            antarctic_bounds: (lat_min, lon_min, lat_max, lon_max) in degrees
            depth_level: Depth in meters (default 0.5m for near-surface)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.antarctic_bounds = antarctic_bounds
        self.depth_level = depth_level

        # Product and variable specifications
        # GLORYS12V1: 1/12° resolution, 1993-present
        self.product_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

        self.variables = {
            'current_u': {
                'name': 'uo',  # Eastward velocity
                'standard_name': 'eastward_sea_water_velocity',
                'units': 'm/s',
                'description': 'Eastward ocean current velocity (U component)'
            },
            'current_v': {
                'name': 'vo',  # Northward velocity
                'standard_name': 'northward_sea_water_velocity',
                'units': 'm/s',
                'description': 'Northward ocean current velocity (V component)'
            }
        }

        # Check if copernicusmarine is available
        try:
            result = subprocess.run(
                ['copernicusmarine', '--version'],
                capture_output=True,
                text=True
            )
            logger.info(f"Copernicus Marine toolbox available: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("copernicusmarine command not found")
            logger.error("Install with: pip install copernicusmarine")
            logger.error("Then login with: copernicusmarine login")
            raise RuntimeError("copernicusmarine toolbox not installed")

    def download_variable_daterange(
        self,
        variable_key: str,
        start_date: datetime,
        end_date: datetime,
        monthly_chunks: bool = True
    ) -> List[Path]:
        """
        Download ocean current component for a date range.

        Args:
            variable_key: 'current_u' or 'current_v'
            start_date: Start date
            end_date: End date (inclusive)
            monthly_chunks: Download in monthly files (recommended)

        Returns:
            List of downloaded file paths
        """
        if variable_key not in self.variables:
            raise ValueError(f"Unknown variable: {variable_key}. "
                           f"Choose from {list(self.variables.keys())}")

        var_info = self.variables[variable_key]
        logger.info(f"Downloading {var_info['description']} from {start_date.date()} to {end_date.date()}")

        downloaded_files = []

        if monthly_chunks:
            # Download month by month
            current_date = start_date.replace(day=1)

            while current_date <= end_date:
                # Get last day of current month
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    next_month = current_date.replace(month=current_date.month + 1)

                month_end = next_month - timedelta(days=1)

                # Don't go past end_date
                if month_end > end_date:
                    month_end = end_date

                # Download this month
                file_path = self._download_variable_month(
                    variable_key,
                    current_date,
                    month_end
                )

                if file_path:
                    downloaded_files.append(file_path)

                current_date = next_month
        else:
            # Download entire range
            file_path = self._download_variable_period(
                variable_key,
                start_date,
                end_date
            )

            if file_path:
                downloaded_files.append(file_path)

        logger.info(f"Downloaded {len(downloaded_files)} files for {variable_key}")
        return downloaded_files

    def _download_variable_month(
        self,
        variable_key: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Path]:
        """Download variable for a single month using copernicusmarine CLI."""
        var_info = self.variables[variable_key]
        var_name = var_info['name']

        # Generate output filename
        year = start_date.year
        month = start_date.month
        output_file = self.output_dir / f"cmems_{variable_key}_{year}{month:02d}.nc"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"File already exists: {output_file.name}")
            return output_file

        # Build command
        lat_min, lon_min, lat_max, lon_max = self.antarctic_bounds

        # Convert longitude from 0-360 to -180-180 if needed by CMEMS
        if lon_max > 180:
            lon_min_180 = lon_min if lon_min <= 180 else lon_min - 360
            lon_max_180 = lon_max if lon_max <= 180 else lon_max - 360
        else:
            lon_min_180 = lon_min
            lon_max_180 = lon_max

        cmd = [
            'copernicusmarine', 'subset',
            '--dataset-id', self.product_id,
            '--variable', var_name,
            '--start-datetime', start_date.strftime('%Y-%m-%d'),
            '--end-datetime', end_date.strftime('%Y-%m-%d'),
            '--minimum-latitude', str(lat_min),
            '--maximum-latitude', str(lat_max),
            '--minimum-longitude', str(lon_min_180),
            '--maximum-longitude', str(lon_max_180),
            '--minimum-depth', str(self.depth_level),
            '--maximum-depth', str(self.depth_level),
            '--output-filename', output_file.name,
            '--output-directory', str(self.output_dir),
            '--force-download'
        ]

        try:
            logger.info(f"Requesting {variable_key} for {year}-{month:02d}")
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Successfully downloaded: {output_file.name}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {variable_key} for {year}-{month:02d}")
            logger.error(f"Error: {e.stderr}")
            if output_file.exists():
                output_file.unlink()
            return None

    def _download_variable_period(
        self,
        variable_key: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Path]:
        """Download variable for an arbitrary period."""
        var_info = self.variables[variable_key]
        var_name = var_info['name']

        # Generate output filename
        output_file = self.output_dir / f"cmems_{variable_key}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"File already exists: {output_file.name}")
            return output_file

        # Build command (similar to monthly)
        lat_min, lon_min, lat_max, lon_max = self.antarctic_bounds

        lon_min_180 = lon_min if lon_min <= 180 else lon_min - 360
        lon_max_180 = lon_max if lon_max <= 180 else lon_max - 360

        cmd = [
            'copernicusmarine', 'subset',
            '--dataset-id', self.product_id,
            '--variable', var_name,
            '--start-datetime', start_date.strftime('%Y-%m-%d'),
            '--end-datetime', end_date.strftime('%Y-%m-%d'),
            '--minimum-latitude', str(lat_min),
            '--maximum-latitude', str(lat_max),
            '--minimum-longitude', str(lon_min_180),
            '--maximum-longitude', str(lon_max_180),
            '--minimum-depth', str(self.depth_level),
            '--maximum-depth', str(self.depth_level),
            '--output-filename', output_file.name,
            '--output-directory', str(self.output_dir),
            '--force-download'
        ]

        try:
            logger.info(f"Requesting {variable_key} for {(end_date - start_date).days + 1} days")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Successfully downloaded: {output_file.name}")
            return output_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {variable_key}")
            logger.error(f"Error: {e.stderr}")
            if output_file.exists():
                output_file.unlink()
            return None

    def merge_monthly_files(
        self,
        variable_key: str,
        file_paths: List[Path],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Merge monthly CMEMS files into a single dataset.

        Args:
            variable_key: Variable identifier
            file_paths: List of monthly NetCDF files
            output_path: Optional output path for merged file

        Returns:
            Path to merged file
        """
        if not file_paths:
            raise ValueError("No files to merge")

        if output_path is None:
            output_path = self.output_dir / f"cmems_{variable_key}_merged.nc"

        logger.info(f"Merging {len(file_paths)} files for {variable_key}")

        # Load and concatenate datasets
        datasets = []
        for file_path in sorted(file_paths):
            ds = xr.open_dataset(file_path)
            datasets.append(ds)

        # Concatenate along time dimension
        merged = xr.concat(datasets, dim='time')

        # Sort by time
        merged = merged.sortby('time')

        # Save
        merged.to_netcdf(output_path)
        logger.info(f"Merged dataset saved to: {output_path}")

        # Close datasets
        for ds in datasets:
            ds.close()

        return output_path

    def check_coverage(self, file_path: Path) -> Dict:
        """
        Check temporal and spatial coverage of a downloaded file.

        Args:
            file_path: Path to NetCDF file

        Returns:
            Dictionary with coverage information
        """
        ds = xr.open_dataset(file_path)

        # Get variable name (should be 'uo' or 'vo')
        var_names = list(ds.data_vars.keys())
        if not var_names:
            raise ValueError(f"No data variables found in {file_path}")

        var_name = var_names[0]

        # Temporal coverage
        time_coord = ds['time']
        time_start = str(time_coord.values[0])
        time_end = str(time_coord.values[-1])
        n_timesteps = len(time_coord)

        # Spatial coverage
        lat = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
        lon = ds['longitude'].values if 'longitude' in ds else ds['lon'].values

        # Depth
        if 'depth' in ds:
            depth = ds['depth'].values
        else:
            depth = [self.depth_level]

        # Missing data
        data = ds[var_name].values
        n_total = data.size
        n_missing = np.isnan(data).sum()
        missing_fraction = n_missing / n_total

        coverage = {
            'file': str(file_path.name),
            'variable': var_name,
            'time_start': time_start,
            'time_end': time_end,
            'n_timesteps': int(n_timesteps),
            'lat_range': [float(lat.min()), float(lat.max())],
            'lon_range': [float(lon.min()), float(lon.max())],
            'spatial_shape': [len(lat), len(lon)],
            'depth_levels': depth.tolist() if hasattr(depth, 'tolist') else [float(depth)],
            'missing_data': {
                'n_missing': int(n_missing),
                'fraction': float(missing_fraction)
            }
        }

        ds.close()

        return coverage

    def get_variable_info(self) -> Dict:
        """Return information about available variables."""
        return self.variables.copy()


def download_all_ocean_currents(
    output_dir: str,
    start_date: str,
    end_date: str
) -> Dict[str, List[Path]]:
    """
    Convenience function to download both ocean current components.

    Args:
        output_dir: Output directory
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping variable keys to lists of downloaded files
    """
    downloader = CopernicusMarineDownloader(output_dir)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    results = {}

    for var_key in ['current_u', 'current_v']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {var_key}")
        logger.info(f"{'='*60}")

        files = downloader.download_variable_daterange(
            var_key,
            start,
            end,
            monthly_chunks=True
        )

        results[var_key] = files

        # Check coverage of first file
        if files:
            coverage = downloader.check_coverage(files[0])
            logger.info(f"Coverage check for {files[0].name}:")
            logger.info(f"  Time: {coverage['time_start']} to {coverage['time_end']}")
            logger.info(f"  Spatial: {coverage['spatial_shape']}")
            logger.info(f"  Depth: {coverage['depth_levels']}")
            logger.info(f"  Missing data: {coverage['missing_data']['fraction']:.2%}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine ocean current data"
    )
    parser.add_argument('--output-dir', type=str, default='data/raw/copernicus',
                       help='Output directory')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--depth', type=float, default=0.5,
                       help='Depth level in meters (default: 0.5)')

    args = parser.parse_args()

    # Create downloader with specified depth
    downloader = CopernicusMarineDownloader(args.output_dir, depth_level=args.depth)

    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Download both U and V components
    results = {}
    for var_key in ['current_u', 'current_v']:
        files = downloader.download_variable_daterange(var_key, start, end)
        results[var_key] = files

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for var_key, files in results.items():
        print(f"{var_key}: {len(files)} files downloaded")
