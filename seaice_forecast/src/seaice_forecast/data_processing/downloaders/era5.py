"""
ERA5 Reanalysis Downloader for Atmospheric and Sea Surface Temperature Variables.

Downloads unified 4-variable monthly chunks from ECMWF Copernicus Climate Data Store (CDS):
- 10m eastward wind (u10)
- 10m northward wind (v10)
- 2m air temperature (t2m)
- Sea surface temperature (sst)

Area: Southern Ocean south of 50°S ([-50, -180, -90, 180]).
Temporal aggregation: Computes daily means from 6-hourly or specified timestamps.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)

ERA5_DATASET = "reanalysis-era5-single-levels"
ANTARCTIC_AREA = [-50, -180, -90, 180]  # [North, West, South, East]

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
]


class ERA5Downloader:
    """Production downloader for ERA5 atmospheric and SST forcing data."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/raw/era5")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._init_client()

    def _init_client(self):
        """Initialize and verify CDS API client."""
        try:
            import cdsapi
            client = cdsapi.Client()
            return client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CDS API client: {e}. "
                "Ensure ~/.cdsapirc exists with valid URL and personal access token."
            )

    def download_month(
        self,
        year: int,
        month: int,
        output_file: Optional[Path] = None,
        days: Optional[List[int]] = None,
        daily_aggregation: bool = True,
        times: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> Path:
        """
        Download a unified monthly (or custom day range) chunk of all 4 variables.

        Args:
            year: year (e.g. 2021)
            month: month (1-12)
            output_file: target NetCDF file path
            days: optional list of day integers (1-31) to restrict query
            daily_aggregation: if True, computes daily mean and saves daily steps
            times: time steps to query (defaults to 6-hourly ['00:00', '06:00', '12:00', '18:00'])
            max_retries: number of download retries

        Returns:
            Path to verified NetCDF file containing daily fields
        """
        import calendar
        _, num_days = calendar.monthrange(year, month)
        if days is None:
            day_strs = [f"{d:02d}" for d in range(1, num_days + 1)]
        else:
            day_strs = [f"{d:02d}" for d in sorted(days)]

        if times is None:
            # 6-hourly provides high fidelity daily means
            times = ["00:00", "06:00", "12:00", "18:00"]

        if output_file is None:
            output_file = self.output_dir / f"era5_forcing_{year}{month:02d}.nc"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists() and output_file.stat().st_size > 100000:
            logger.debug(f"ERA5 file already exists: {output_file}")
            return output_file

        tmp_raw = output_file.with_suffix(".raw.nc")
        tmp_final = output_file.with_suffix(".tmp.nc")

        request = {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": day_strs,
            "time": times,
            "format": "netcdf",
            "area": ANTARCTIC_AREA,
        }

        logger.info(
            f"Submitting CDS request for ERA5 {year}-{month:02d} ({len(day_strs)} days, 4 variables)..."
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self.client.retrieve(ERA5_DATASET, request, str(tmp_raw))

                if not tmp_raw.exists():
                    raise FileNotFoundError(f"CDS API completed but output file {tmp_raw} not found.")

                # Validate and aggregate to daily means
                with xr.open_dataset(tmp_raw) as ds:
                    # Identify time coordinate
                    t_coord = "valid_time" if "valid_time" in ds.coords else "time"

                    # Check required variables
                    for expected in ["u10", "v10", "t2m"]:
                        if expected not in ds:
                            raise ValueError(f"Missing expected variable {expected} in ERA5 response: {list(ds.data_vars)}")

                    # Daily mean aggregation if requested
                    if daily_aggregation and len(times) > 1:
                        logger.info(f"Computing daily mean from {len(times)} daily timesteps...")
                        ds_daily = ds.resample({t_coord: "1D"}).mean(dim=t_coord)
                        ds_daily.to_netcdf(tmp_final)
                    else:
                        ds.to_netcdf(tmp_final)

                # Clean up raw download
                if tmp_raw.exists():
                    tmp_raw.unlink()

                tmp_final.replace(output_file)
                logger.info(f"ERA5 monthly chunk saved successfully: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")
                return output_file

            except Exception as e:
                last_err = e
                for f in [tmp_raw, tmp_final]:
                    if f.exists():
                        f.unlink()
                logger.warning(f"ERA5 download attempt {attempt} failed: {e}")
                import time
                time.sleep(5 * attempt)

        raise RuntimeError(
            f"Failed to download real ERA5 data for {year}-{month:02d} after {max_retries} attempts: {last_err}. "
            "Synthetic fallback is strictly disabled."
        )
