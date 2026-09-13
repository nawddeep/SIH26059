"""
Download real NOAA/NSIDC Sea Ice Concentration CDR v6 data (G02202).

Antarctic (South) passive microwave sea ice concentration at 25km resolution
on the polar stereographic grid (EPSG:3412, shape 332x316).

Primary source: NOAA NSIDC open data archive
  https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/{year}/sic_pss25_{YYYYMMDD}_{sensor}_v06r00.nc

Sensors:
  - F17: SSMIS on DMSP F17 (2006-2023)
  - F18: SSMIS on DMSP F18 (2010-present)
"""

import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import logging
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NSIDCDownloader:
    """Production downloader for NSIDC Sea Ice Concentration CDR v6."""

    def __init__(self, output_dir: Optional[Path] = None, hemisphere: str = "south"):
        self.output_dir = Path(output_dir) if output_dir else Path("data/raw/nsidc")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hemisphere = hemisphere
        self.base_url = f"https://noaadata.apps.nsidc.org/NOAA/G02202_V6/{hemisphere}/daily"

        # Ordered candidate filename patterns
        # Patterns use {date} (YYYYMMDD) and {sensor} (F17, F18, etc.)
        self.sensors = ["F17", "F18", "f17", "f18"]
        self.pattern_templates = [
            "sic_pss25_{date}_{sensor}_v06r00.nc",
            "sic_pss25_{date}_{sensor}_v06r01.nc",
            "seaice_conc_daily_sh_{date}_{sensor}_v06r00.nc",
            "seaice_conc_daily_sh_{date}_{sensor}_v06r01.nc",
        ]

    def get_candidate_urls(self, date: datetime) -> List[Tuple[str, str]]:
        """
        Generate candidate (url, filename) pairs for a date in priority order.
        """
        date_str = date.strftime("%Y%m%d")
        year = date.year
        candidates = []
        for sensor in self.sensors:
            for template in self.pattern_templates:
                fname = template.format(date=date_str, sensor=sensor)
                url = f"{self.base_url}/{year}/{fname}"
                candidates.append((url, fname))
        return candidates

    def download_file(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """
        Download a file with exponential backoff and atomic rename.
        """
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 404:
                    return False
                response.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

                # Validate downloaded NetCDF
                try:
                    with xr.open_dataset(tmp_path) as ds:
                        var_found = any("conc" in v.lower() for v in ds.data_vars)
                        if not var_found:
                            raise ValueError(f"Downloaded file has no SIC variable: {list(ds.data_vars.keys())}")
                except Exception as val_err:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    logger.warning(f"File validation failed for {url} (attempt {attempt}): {val_err}")
                    time.sleep(2 ** attempt)
                    continue

                # Atomic rename
                tmp_path.replace(output_path)
                return True

            except (requests.RequestException, IOError) as e:
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt == max_retries:
                    logger.debug(f"Failed to download {url} after {max_retries} attempts: {e}")
                    return False
                time.sleep(2 ** attempt)

        return False

    def download_date(self, date: datetime, target_dir: Optional[Path] = None) -> Path:
        """
        Download data for a specific date, trying candidate filenames.

        Raises:
            FileNotFoundError: If real data cannot be retrieved from any source.
                               NO synthetic fallback is ever performed.
        """
        save_dir = Path(target_dir) if target_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        date_str = date.strftime("%Y%m%d")

        # Check if already downloaded under any candidate name
        candidates = self.get_candidate_urls(date)
        for _, fname in candidates:
            existing = save_dir / fname
            if existing.exists() and existing.stat().st_size > 50000:
                return existing

        # Attempt download across candidates
        for url, fname in candidates:
            out_path = save_dir / fname
            if self.download_file(url, out_path):
                logger.debug(f"Downloaded NSIDC SIC for {date_str}: {fname}")
                return out_path

        raise FileNotFoundError(
            f"Failed to download real NSIDC SIC data for {date.strftime('%Y-%m-%d')} from NOAA archive. "
            f"Attempted {len(candidates)} candidate URLs. Synthetic fallback is strictly disabled."
        )

    def download_month(self, year: int, month: int, target_dir: Optional[Path] = None) -> List[Path]:
        """
        Download all days for a specified month.
        """
        import calendar
        _, num_days = calendar.monthrange(year, month)
        downloaded = []
        for day in range(1, num_days + 1):
            d = datetime(year, month, day)
            path = self.download_date(d, target_dir=target_dir)
            downloaded.append(path)
        return downloaded

    def check_availability(self, date: datetime) -> bool:
        """Check availability via HTTP HEAD without downloading."""
        for url, _ in self.get_candidate_urls(date):
            try:
                r = requests.head(url, timeout=10)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                continue
        return False
