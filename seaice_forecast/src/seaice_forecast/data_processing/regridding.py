"""
Regrid environmental variables to match SIC polar stereographic grid.

All environmental forcing variables (ERA5, Copernicus Marine) must be
resampled to the same grid as the NSIDC SIC data for model input.

Reference grid: NSIDC PS25 (25km Antarctic polar stereographic)
- Shape: 316 × 332
- Projection: EPSG:3031 (Antarctic Polar Stereographic)

Regridding methods:
- Wind U/V: Bilinear interpolation (smooth continuous field)
- Air temperature: Bilinear interpolation (smooth continuous field)
- SST: Bilinear interpolation with ocean mask
- Ocean currents U/V: Bilinear interpolation with ocean mask

Missing data handling: Explicitly documented per variable
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
from scipy.interpolate import griddata, RegularGridInterpolator
from pyproj import Transformer
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentalRegridder:
    """
    Regrid environmental variables to NSIDC SIC grid.

    Handles spatial alignment, temporal alignment, and missing data
    for all Phase 2 environmental forcing variables.
    """

    def __init__(
        self,
        sic_reference_file: Path,
        land_ocean_mask: Optional[np.ndarray] = None
    ):
        """
        Initialize regridder with SIC reference grid.

        Args:
            sic_reference_file: Path to a reference SIC NetCDF file
            land_ocean_mask: Optional land/ocean mask (0=land, 1=ocean)
        """
        self.sic_file = Path(sic_reference_file)

        # Load SIC reference grid
        logger.info(f"Loading SIC reference grid from: {self.sic_file.name}")
        sic_ds = xr.open_dataset(self.sic_file)

        # Get grid information
        self.target_shape = sic_ds.dims['y'], sic_ds.dims['x']  # (316, 332)

        # Get projection coordinates
        self.target_x = sic_ds['x'].values
        self.target_y = sic_ds['y'].values

        # Create 2D coordinate grids
        self.target_X, self.target_Y = np.meshgrid(self.target_x, self.target_y)

        # Get projection info (EPSG:3031 for Antarctic PS)
        self.target_crs = 'EPSG:3031'

        logger.info(f"Target grid shape: {self.target_shape}")
        logger.info(f"Target CRS: {self.target_crs}")

        sic_ds.close()

        # Store mask
        self.mask = land_ocean_mask
        if self.mask is not None:
            logger.info(f"Land/ocean mask loaded: {np.sum(self.mask == 1)} ocean cells, "
                       f"{np.sum(self.mask == 0)} land cells")

    def regrid_era5_variable(
        self,
        era5_file: Path,
        variable_name: str,
        output_file: Path,
        method: str = 'bilinear',
        fill_missing: Optional[float] = None
    ) -> Tuple[Path, Dict]:
        """
        Regrid ERA5 variable from lat/lon to polar stereographic grid.

        Args:
            era5_file: Path to ERA5 NetCDF file
            variable_name: Variable name in file (e.g., 'u10', 'v10', 't2m', 'sst')
            output_file: Output path for regridded file
            method: Interpolation method ('bilinear', 'nearest')
            fill_missing: Value to fill missing data (None = leave as NaN)

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        logger.info(f"Regridding ERA5 variable: {variable_name}")
        logger.info(f"Input: {era5_file.name}")
        logger.info(f"Method: {method}")

        # Load ERA5 data
        ds = xr.open_dataset(era5_file)

        # Identify coordinate names (ERA5 uses 'latitude'/'longitude' or 'lat'/'lon')
        lat_name = 'latitude' if 'latitude' in ds else 'lat'
        lon_name = 'longitude' if 'longitude' in ds else 'lon'

        # Get actual variable name if not exact
        if variable_name not in ds:
            # Try to find it
            possible_names = [k for k in ds.data_vars.keys()]
            if len(possible_names) == 1:
                actual_var_name = possible_names[0]
                logger.info(f"Variable name '{variable_name}' not found, using '{actual_var_name}'")
                variable_name = actual_var_name
            else:
                raise ValueError(f"Variable '{variable_name}' not found. Available: {possible_names}")

        # Get source coordinates
        source_lat = ds[lat_name].values
        source_lon = ds[lon_name].values

        # Convert lon from 0-360 to -180-180 if needed
        if source_lon.max() > 180:
            source_lon = np.where(source_lon > 180, source_lon - 360, source_lon)

        # Create 2D grids
        source_LON, source_LAT = np.meshgrid(source_lon, source_lat)

        # Transform source lat/lon to target projection (polar stereographic)
        transformer = Transformer.from_crs('EPSG:4326', self.target_crs, always_xy=True)
        source_X, source_Y = transformer.transform(source_LON, source_LAT)

        # Get time dimension
        time = ds['time'].values
        n_times = len(time)

        # Prepare output array
        output_data = np.zeros((n_times, *self.target_shape), dtype=np.float32)

        # Count missing data
        missing_before = 0
        missing_after = 0

        # Regrid each time step
        logger.info(f"Regridding {n_times} time steps...")

        for t in range(n_times):
            # Get data for this timestep
            source_data = ds[variable_name].isel(time=t).values

            # Track missing data before regridding
            missing_before += np.isnan(source_data).sum()

            # Flatten for interpolation
            valid_mask = ~np.isnan(source_data)

            if valid_mask.sum() == 0:
                logger.warning(f"Time step {t}: All data is NaN")
                output_data[t] = np.nan
                missing_after += self.target_shape[0] * self.target_shape[1]
                continue

            source_points = np.column_stack([
                source_X[valid_mask].ravel(),
                source_Y[valid_mask].ravel()
            ])
            source_values = source_data[valid_mask].ravel()

            target_points = np.column_stack([
                self.target_X.ravel(),
                self.target_Y.ravel()
            ])

            # Interpolate
            if method == 'bilinear':
                # Use griddata with linear interpolation
                output_flat = griddata(
                    source_points,
                    source_values,
                    target_points,
                    method='linear',
                    fill_value=np.nan
                )
            elif method == 'nearest':
                output_flat = griddata(
                    source_points,
                    source_values,
                    target_points,
                    method='nearest'
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            output_data[t] = output_flat.reshape(self.target_shape)

            # Track missing data after regridding
            missing_after += np.isnan(output_data[t]).sum()

        # Apply fill value if specified
        if fill_missing is not None:
            output_data[np.isnan(output_data)] = fill_missing
            logger.info(f"Filled missing values with {fill_missing}")

        # Apply land mask if available (mask out land pixels)
        if self.mask is not None:
            land_mask = self.mask == 0
            for t in range(n_times):
                output_data[t][land_mask] = np.nan

        # Create output dataset
        output_ds = xr.Dataset(
            {
                variable_name: (['time', 'y', 'x'], output_data),
            },
            coords={
                'time': time,
                'y': self.target_y,
                'x': self.target_x
            },
            attrs={
                'source_file': str(era5_file.name),
                'regrid_method': method,
                'target_crs': self.target_crs,
                'target_shape': str(self.target_shape),
                'fill_missing': str(fill_missing) if fill_missing is not None else 'None'
            }
        )

        # Add variable attributes from source
        if variable_name in ds:
            for attr_name in ['units', 'long_name', 'standard_name']:
                if attr_name in ds[variable_name].attrs:
                    output_ds[variable_name].attrs[attr_name] = ds[variable_name].attrs[attr_name]

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_ds.to_netcdf(output_file)
        logger.info(f"Saved regridded data to: {output_file.name}")

        # Calculate statistics
        source_cells = source_data.size * n_times
        target_cells = self.target_shape[0] * self.target_shape[1] * n_times

        metadata = {
            'variable': variable_name,
            'source_file': str(era5_file.name),
            'output_file': str(output_file.name),
            'source_shape': source_data.shape,
            'target_shape': self.target_shape,
            'n_timesteps': n_times,
            'method': method,
            'missing_data': {
                'source': {
                    'count': int(missing_before),
                    'fraction': float(missing_before / source_cells)
                },
                'target': {
                    'count': int(missing_after),
                    'fraction': float(missing_after / target_cells)
                }
            },
            'fill_value': fill_missing
        }

        logger.info(f"Missing data - Source: {metadata['missing_data']['source']['fraction']:.2%}, "
                   f"Target: {metadata['missing_data']['target']['fraction']:.2%}")

        ds.close()
        output_ds.close()

        return output_file, metadata

    def regrid_copernicus_variable(
        self,
        copernicus_file: Path,
        variable_name: str,
        output_file: Path,
        method: str = 'bilinear',
        fill_missing: Optional[float] = None
    ) -> Tuple[Path, Dict]:
        """
        Regrid Copernicus Marine variable to polar stereographic grid.

        Similar to ERA5 regridding but handles CMEMS-specific structure.

        Args:
            copernicus_file: Path to CMEMS NetCDF file
            variable_name: Variable name ('uo' or 'vo')
            output_file: Output path
            method: Interpolation method
            fill_missing: Value to fill missing data

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        logger.info(f"Regridding Copernicus Marine variable: {variable_name}")
        logger.info(f"Input: {copernicus_file.name}")

        # Load data
        ds = xr.open_dataset(copernicus_file)

        # CMEMS uses 'latitude'/'longitude'
        lat_name = 'latitude' if 'latitude' in ds else 'lat'
        lon_name = 'longitude' if 'longitude' in ds else 'lon'

        # Get source coordinates
        source_lat = ds[lat_name].values
        source_lon = ds[lon_name].values

        # Convert lon if needed
        if source_lon.max() > 180:
            source_lon = np.where(source_lon > 180, source_lon - 360, source_lon)

        # If there's a depth dimension, select surface level
        if 'depth' in ds[variable_name].dims:
            logger.info("Selecting surface depth level")
            ds_surface = ds[variable_name].isel(depth=0)
        else:
            ds_surface = ds[variable_name]

        # Create 2D grids
        source_LON, source_LAT = np.meshgrid(source_lon, source_lat)

        # Transform to polar stereographic
        transformer = Transformer.from_crs('EPSG:4326', self.target_crs, always_xy=True)
        source_X, source_Y = transformer.transform(source_LON, source_LAT)

        # Get time
        time = ds['time'].values
        n_times = len(time)

        # Prepare output
        output_data = np.zeros((n_times, *self.target_shape), dtype=np.float32)

        missing_before = 0
        missing_after = 0

        logger.info(f"Regridding {n_times} time steps...")

        for t in range(n_times):
            # Get data
            if 'depth' in ds[variable_name].dims:
                source_data = ds[variable_name].isel(time=t, depth=0).values
            else:
                source_data = ds[variable_name].isel(time=t).values

            missing_before += np.isnan(source_data).sum()

            # Interpolate
            valid_mask = ~np.isnan(source_data)

            if valid_mask.sum() == 0:
                logger.warning(f"Time step {t}: All data is NaN")
                output_data[t] = np.nan
                missing_after += self.target_shape[0] * self.target_shape[1]
                continue

            source_points = np.column_stack([
                source_X[valid_mask].ravel(),
                source_Y[valid_mask].ravel()
            ])
            source_values = source_data[valid_mask].ravel()

            target_points = np.column_stack([
                self.target_X.ravel(),
                self.target_Y.ravel()
            ])

            if method == 'bilinear':
                output_flat = griddata(
                    source_points,
                    source_values,
                    target_points,
                    method='linear',
                    fill_value=np.nan
                )
            elif method == 'nearest':
                output_flat = griddata(
                    source_points,
                    source_values,
                    target_points,
                    method='nearest'
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            output_data[t] = output_flat.reshape(self.target_shape)
            missing_after += np.isnan(output_data[t]).sum()

        # Fill missing if specified
        if fill_missing is not None:
            output_data[np.isnan(output_data)] = fill_missing

        # Apply ocean mask (currents only exist in ocean)
        if self.mask is not None:
            land_mask = self.mask == 0
            for t in range(n_times):
                output_data[t][land_mask] = np.nan

        # Create output dataset
        output_ds = xr.Dataset(
            {
                variable_name: (['time', 'y', 'x'], output_data),
            },
            coords={
                'time': time,
                'y': self.target_y,
                'x': self.target_x
            },
            attrs={
                'source_file': str(copernicus_file.name),
                'regrid_method': method,
                'target_crs': self.target_crs
            }
        )

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_ds.to_netcdf(output_file)
        logger.info(f"Saved to: {output_file.name}")

        # Metadata
        source_cells = source_data.size * n_times
        target_cells = self.target_shape[0] * self.target_shape[1] * n_times

        metadata = {
            'variable': variable_name,
            'source_file': str(copernicus_file.name),
            'output_file': str(output_file.name),
            'n_timesteps': n_times,
            'method': method,
            'missing_data': {
                'source': {
                    'count': int(missing_before),
                    'fraction': float(missing_before / source_cells)
                },
                'target': {
                    'count': int(missing_after),
                    'fraction': float(missing_after / target_cells)
                }
            }
        }

        logger.info(f"Missing data - Source: {metadata['missing_data']['source']['fraction']:.2%}, "
                   f"Target: {metadata['missing_data']['target']['fraction']:.2%}")

        ds.close()
        output_ds.close()

        return output_file, metadata

    def temporal_align_to_sic(
        self,
        env_file: Path,
        sic_dates: List[np.datetime64],
        variable_name: str,
        output_file: Path,
        aggregation_method: str = 'mean'
    ) -> Tuple[Path, Dict]:
        """
        Align environmental variable temporally to match SIC daily timestamps.

        Handles cases where environmental data has different temporal frequency
        (e.g., ERA5 hourly aggregated to daily).

        Args:
            env_file: Path to regridded environmental variable file
            sic_dates: List of daily dates from SIC dataset
            variable_name: Variable name
            output_file: Output path
            aggregation_method: How to aggregate to daily ('mean', 'min', 'max')

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        logger.info(f"Temporally aligning {variable_name} to SIC dates")
        logger.info(f"Aggregation method: {aggregation_method}")

        # Load environmental data
        ds = xr.open_dataset(env_file)

        # Convert SIC dates to pandas datetime
        sic_dates_pd = [np.datetime64(d, 'D') for d in sic_dates]

        # Resample to daily if needed
        if aggregation_method == 'mean':
            daily_ds = ds.resample(time='1D').mean()
        elif aggregation_method == 'min':
            daily_ds = ds.resample(time='1D').min()
        elif aggregation_method == 'max':
            daily_ds = ds.resample(time='1D').max()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Select only dates that match SIC
        aligned_ds = daily_ds.sel(time=sic_dates_pd, method='nearest')

        # Save
        output_file.parent.mkdir(parents=True, exist_ok=True)
        aligned_ds.to_netcdf(output_file)

        metadata = {
            'variable': variable_name,
            'source_timesteps': len(ds['time']),
            'aligned_timesteps': len(aligned_ds['time']),
            'aggregation_method': aggregation_method,
            'date_range': {
                'start': str(aligned_ds['time'].values[0]),
                'end': str(aligned_ds['time'].values[-1])
            }
        }

        logger.info(f"Aligned {metadata['source_timesteps']} -> {metadata['aligned_timesteps']} timesteps")

        ds.close()
        aligned_ds.close()

        return output_file, metadata


def test_regridder():
    """Test regridding functionality."""
    # This would need actual data files to run
    logger.info("Regridder test placeholder - requires actual data files")


if __name__ == "__main__":
    test_regridder()
