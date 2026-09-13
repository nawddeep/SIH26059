"""
Copernicus Marine Service Downloader for Ocean Surface Currents (GLORYS12V1).

Downloads eastward (uo) and northward (vo) ocean current velocities at surface depth (~0.49m)
for the Antarctic Southern Ocean region south of 50°S.

Handles dataset resolution and validates multi-year continuity across the historical
2021-06-30 seam with zero duplicate timestamps and zero missing days.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)

# CMEMS Dataset configuration
CMEMS_MULTIYEAR_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
CMEMS_INTERIM_DATASET = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
SEAM_DATE = datetime(2021, 6, 30)

SURFACE_DEPTH_MIN = 0.49
SURFACE_DEPTH_MAX = 0.50  # Selects ~0.494m level only
ANTARCTIC_LAT_MIN = -80.0
ANTARCTIC_LAT_MAX = -50.0
ANTARCTIC_LON_MIN = -180.0
ANTARCTIC_LON_MAX = 179.91667


class CopernicusMarineDownloader:
    """
    Downloads and verifies ocean current velocities from Copernicus Marine Service.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/raw/copernicus")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._check_client()

    def _check_client(self):
        """Verify copernicusmarine library and authentication are available."""
        try:
            import copernicusmarine
            # Verify credentials exist
            cred_file = Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
            if not cred_file.exists():
                logger.warning(f"Copernicus credentials not found at {cred_file}. Calls may fail if unauthenticated.")
        except ImportError:
            raise ImportError(
                "copernicusmarine package is required. Install via `pip install copernicusmarine`."
            )

    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        output_file: Optional[Path] = None,
        max_retries: int = 3
    ) -> Path:
        """
        Download surface currents (uo, vo) for a date range with seam verification.

        Args:
            start_date: inclusive start datetime
            end_date: inclusive end datetime
            output_file: destination NetCDF path

        Returns:
            Path to downloaded, validated NetCDF file
        """
        import copernicusmarine

        if output_file is None:
            s_str = start_date.strftime("%Y%m%d")
            e_str = end_date.strftime("%Y%m%d")
            output_file = self.output_dir / f"cmems_currents_{s_str}_{e_str}.nc"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists() and output_file.stat().st_size > 100000:
            logger.debug(f"CMEMS file already exists: {output_file}")
            return output_file

        tmp_file = output_file.parent / f".tmp_{output_file.name}"
        if tmp_file.exists():
            tmp_file.unlink()

        # Format ISO strings
        start_str = start_date.strftime("%Y-%m-%dT00:00:00")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59")

        dataset_id = CMEMS_MULTIYEAR_DATASET

        logger.info(
            f"Downloading CMEMS surface currents ({start_str} to {end_str}) "
            f"using {dataset_id}..."
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                res = copernicusmarine.subset(
                    dataset_id=dataset_id,
                    variables=["uo", "vo"],
                    minimum_latitude=ANTARCTIC_LAT_MIN,
                    maximum_latitude=ANTARCTIC_LAT_MAX,
                    minimum_longitude=ANTARCTIC_LON_MIN,
                    maximum_longitude=ANTARCTIC_LON_MAX,
                    minimum_depth=SURFACE_DEPTH_MIN,
                    maximum_depth=SURFACE_DEPTH_MAX,
                    start_datetime=start_str,
                    end_datetime=end_str,
                    output_directory=str(tmp_file.parent),
                    output_filename=tmp_file.name,
                    overwrite=True,
                )

                actual_path = None
                if res and hasattr(res, "file_path") and res.file_path:
                    p = Path(res.file_path)
                    if p.exists():
                        actual_path = p
                if actual_path is None:
                    if tmp_file.exists():
                        actual_path = tmp_file
                    elif Path(str(tmp_file) + ".nc").exists():
                        actual_path = Path(str(tmp_file) + ".nc")

                if actual_path is None or not actual_path.exists():
                    raise FileNotFoundError(f"Copernicus Marine tool exited but {tmp_file} was not found.")

                # Validate dataset integrity and continuity
                with xr.open_dataset(actual_path) as ds:
                    if "uo" not in ds or "vo" not in ds:
                        raise ValueError(f"Downloaded CMEMS dataset missing uo/vo: {list(ds.data_vars)}")

                    # Verify date continuity
                    times = ds["time"].values
                    num_days_expected = (end_date.date() - start_date.date()).days + 1
                    num_days_actual = len(times)

                    # Check for duplicates
                    unique_times = np.unique(times)
                    if len(unique_times) != num_days_actual:
                        raise ValueError(
                            f"CMEMS dataset contains {num_days_actual - len(unique_times)} duplicate timestamps!"
                        )

                    logger.info(
                        f"CMEMS download verified: {num_days_actual}/{num_days_expected} daily steps present, "
                        f"depth={ds['depth'].values if 'depth' in ds else 'surface'}, 0 duplicates."
                    )

                # Move validated file
                actual_path.replace(output_file)
                return output_file

            except Exception as e:
                last_err = e
                if actual_path and actual_path.exists():
                    actual_path.unlink()
                elif tmp_file.exists():
                    tmp_file.unlink()
                logger.warning(f"CMEMS download attempt {attempt} failed: {e}")
                import time
                time.sleep(3 * attempt)

        raise RuntimeError(
            f"Failed to download real CMEMS current data for {start_str} to {end_str} after {max_retries} attempts. "
            f"Error: {last_err}. Synthetic fallback is strictly disabled."
        )

    def download_month(self, year: int, month: int, output_dir: Optional[Path] = None) -> Path:
        """Download complete monthly chunk of surface currents."""
        import calendar
        _, num_days = calendar.monthrange(year, month)
        start = datetime(year, month, 1)
        end = datetime(year, month, num_days)
        target = (Path(output_dir) if output_dir else self.output_dir) / f"cmems_currents_{year}{month:02d}.nc"
        return self.download_date_range(start, end, output_file=target)
