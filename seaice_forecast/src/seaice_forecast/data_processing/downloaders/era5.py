"""
ERA5 data download module for Antarctic sea-ice forecasting.

Downloads and processes ERA5 reanalysis variables:
- 10m wind U/V components
- 2m air temperature
- Sea surface temperature

Variables are kept as components (U/V) rather than magnitude/direction
to avoid the 0°/360° discontinuity issue.

Data source: Copernicus Climate Data Store (CDS)
Requires: cdsapi library and valid CDS credentials
"""

import cdsapi
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERA5Downloader:
    """
    Download ERA5 reanalysis data for Antarctic region.

    Handles single-level and surface variables needed for
    sea-ice forecasting with environmental forcing.
    """

    def __init__(
        self,
        output_dir: str,
        antarctic_bounds: Tuple[float, float, float, float] = (-90, 0, -50, 360)
    ):
        """
        Initialize ERA5 downloader.

        Args:
            output_dir: Directory to store downloaded files
            antarctic_bounds: (lat_min, lon_min, lat_max, lon_max) in degrees
                             Default covers Antarctic region south of 50°S
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.antarctic_bounds = antarctic_bounds

        # Initialize CDS API client
        try:
            self.client = cdsapi.Client()
            logger.info("CDS API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CDS API client: {e}")
            logger.error("Ensure ~/.cdsapirc exists with valid credentials")
            raise

        # Variable specifications
        self.variables = {
            'wind_u': {
                'name': '10m_u_component_of_wind',
                'dataset': 'reanalysis-era5-single-levels',
                'units': 'm/s',
                'description': '10-metre U wind component'
            },
            'wind_v': {
                'name': '10m_v_component_of_wind',
                'dataset': 'reanalysis-era5-single-levels',
                'units': 'm/s',
                'description': '10-metre V wind component'
            },
            'air_temp': {
                'name': '2m_temperature',
                'dataset': 'reanalysis-era5-single-levels',
                'units': 'K',
                'description': '2-metre temperature'
            },
            'sst': {
                'name': 'sea_surface_temperature',
                'dataset': 'reanalysis-era5-single-levels',
                'units': 'K',
                'description': 'Sea surface temperature'
            }
        }

    def download_variable_daterange(
        self,
        variable_key: str,
        start_date: datetime,
        end_date: datetime,
        time: str = '00:00',
        monthly_chunks: bool = True
    ) -> List[Path]:
        """
        Download a specific ERA5 variable for a date range.

        Args:
            variable_key: Key from self.variables ('wind_u', 'wind_v', 'air_temp', 'sst')
            start_date: Start date
            end_date: End date (inclusive)
            time: Time of day (default '00:00' for daily snapshots)
            monthly_chunks: Download in monthly files (recommended for large ranges)

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
                    month_end,
                    time
                )

                if file_path:
                    downloaded_files.append(file_path)

                current_date = next_month
        else:
            # Download entire range as one file
            file_path = self._download_variable_period(
                variable_key,
                start_date,
                end_date,
                time
            )

            if file_path:
                downloaded_files.append(file_path)

        logger.info(f"Downloaded {len(downloaded_files)} files for {variable_key}")
        return downloaded_files

    def _download_variable_month(
        self,
        variable_key: str,
        start_date: datetime,
        end_date: datetime,
        time: str
    ) -> Optional[Path]:
        """Download variable for a single month."""
        var_info = self.variables[variable_key]

        # Generate output filename
        year = start_date.year
        month = start_date.month
        output_file = self.output_dir / f"era5_{variable_key}_{year}{month:02d}.nc"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"File already exists: {output_file.name}")
            return output_file

        # Generate day list for this month
        days = []
        current = start_date
        while current <= end_date:
            days.append(f"{current.day:02d}")
            current += timedelta(days=1)

        # Build request
        lat_min, lon_min, lat_max, lon_max = self.antarctic_bounds

        request = {
            'product_type': 'reanalysis',
            'variable': var_info['name'],
            'year': f"{year}",
            'month': f"{month:02d}",
            'day': days,
            'time': time,
            'area': [lat_max, lon_min, lat_min, lon_max],  # North, West, South, East
            'format': 'netcdf'
        }

        try:
            logger.info(f"Requesting {variable_key} for {year}-{month:02d} ({len(days)} days)")
            self.client.retrieve(
                var_info['dataset'],
                request,
                str(output_file)
            )
            logger.info(f"Successfully downloaded: {output_file.name}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to download {variable_key} for {year}-{month:02d}: {e}")
            if output_file.exists():
                output_file.unlink()
            return None

    def _download_variable_period(
        self,
        variable_key: str,
        start_date: datetime,
        end_date: datetime,
        time: str
    ) -> Optional[Path]:
        """Download variable for an arbitrary period (not recommended for long ranges)."""
        var_info = self.variables[variable_key]

        # Generate output filename
        output_file = self.output_dir / f"era5_{variable_key}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"File already exists: {output_file.name}")
            return output_file

        # Generate date list
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)

        # Build request
        lat_min, lon_min, lat_max, lon_max = self.antarctic_bounds

        request = {
            'product_type': 'reanalysis',
            'variable': var_info['name'],
            'date': dates,
            'time': time,
            'area': [lat_max, lon_min, lat_min, lon_max],
            'format': 'netcdf'
        }

        try:
            logger.info(f"Requesting {variable_key} for {len(dates)} days")
            self.client.retrieve(
                var_info['dataset'],
                request,
                str(output_file)
            )
            logger.info(f"Successfully downloaded: {output_file.name}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to download {variable_key}: {e}")
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
        Merge monthly ERA5 files into a single dataset.

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
            output_path = self.output_dir / f"era5_{variable_key}_merged.nc"

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

    def get_variable_info(self) -> Dict:
        """Return information about available variables."""
        return self.variables.copy()

    def check_coverage(self, file_path: Path) -> Dict:
        """
        Check temporal and spatial coverage of a downloaded file.

        Args:
            file_path: Path to NetCDF file

        Returns:
            Dictionary with coverage information
        """
        ds = xr.open_dataset(file_path)

        # Get variable name (first data variable)
        var_names = list(ds.data_vars.keys())
        if not var_names:
            raise ValueError(f"No data variables found in {file_path}")

        var_name = var_names[0]

        # Temporal coverage
        time_coord = ds['time']
        time_start = pd.Timestamp(time_coord.values[0])
        time_end = pd.Timestamp(time_coord.values[-1])
        n_timesteps = len(time_coord)

        # Spatial coverage
        lat = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
        lon = ds['longitude'].values if 'longitude' in ds else ds['lon'].values

        # Missing data
        data = ds[var_name].values
        n_total = data.size
        n_missing = np.isnan(data).sum()
        missing_fraction = n_missing / n_total

        coverage = {
            'file': str(file_path.name),
            'variable': var_name,
            'time_start': str(time_start),
            'time_end': str(time_end),
            'n_timesteps': int(n_timesteps),
            'lat_range': [float(lat.min()), float(lat.max())],
            'lon_range': [float(lon.min()), float(lon.max())],
            'spatial_shape': [len(lat), len(lon)],
            'missing_data': {
                'n_missing': int(n_missing),
                'fraction': float(missing_fraction)
            }
        }

        ds.close()

        return coverage


def download_all_era5_variables(
    output_dir: str,
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None
) -> Dict[str, List[Path]]:
    """
    Convenience function to download all ERA5 variables for a date range.

    Args:
        output_dir: Output directory
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        variables: Optional list of variable keys to download
                  (default: all available)

    Returns:
        Dictionary mapping variable keys to lists of downloaded files
    """
    downloader = ERA5Downloader(output_dir)

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    if variables is None:
        variables = ['wind_u', 'wind_v', 'air_temp', 'sst']

    results = {}

    for var_key in variables:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {var_key}")
        logger.info(f"{'='*60}")

        files = downloader.download_variable_daterange(
            var_key,
            start,
            end,
            time='00:00',
            monthly_chunks=True
        )

        results[var_key] = files

        # Check coverage of first file
        if files:
            coverage = downloader.check_coverage(files[0])
            logger.info(f"Coverage check for {files[0].name}:")
            logger.info(f"  Time: {coverage['time_start']} to {coverage['time_end']}")
            logger.info(f"  Spatial: {coverage['spatial_shape']}")
            logger.info(f"  Missing data: {coverage['missing_data']['fraction']:.2%}")

    return results


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Download ERA5 data for Antarctic sea-ice forecasting")
    parser.add_argument('--output-dir', type=str, default='data/raw/era5',
                       help='Output directory')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--variables', nargs='+', default=None,
                       help='Variables to download (default: all)')

    args = parser.parse_args()

    # Download data
    results = download_all_era5_variables(
        args.output_dir,
        args.start_date,
        args.end_date,
        args.variables
    )

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for var_key, files in results.items():
        print(f"{var_key}: {len(files)} files downloaded")
