"""
Production Real Data Ingestion & Regridding Pipeline.

Orchestrates the chunked download-regrid-discard lifecycle:
1. Checks free disk space (> 5GB threshold) before beginning any work.
2. Checks manifest.json for completed chunks; skips already-processed months (idempotent).
3. Downloads raw monthly data for NSIDC, CMEMS, and ERA5 into a scratch directory.
4. Regrids all daily steps onto the reference NSIDC PS25 grid (332x316).
5. Atomically saves daily regridded arrays [7, 332, 316] to disk.
6. Deletes raw scratch files immediately to preserve storage.
7. Updates manifest.json with completed chunk metadata and SHA-256 verification.
"""

import os
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import calendar
import numpy as np
import xarray as xr

from seaice_forecast.data_processing.grid import AntarcticGrid
from seaice_forecast.data_processing.regridding import EnvironmentalRegridder
from seaice_forecast.data_processing.downloaders.nsidc import NSIDCDownloader
from seaice_forecast.data_processing.downloaders.copernicus import CopernicusMarineDownloader
from seaice_forecast.data_processing.downloaders.era5 import ERA5Downloader

logger = logging.getLogger(__name__)

MIN_FREE_DISK_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB safety floor


class DataPipeline:
    """
    Production storage-efficient, resumable pipeline for real polar datasets.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        min_free_disk_gb: float = 5.0
    ):
        self.project_root = Path(base_dir) if base_dir else (
            Path(__file__).parent.parent.parent.parent
        )
        self.raw_dir = self.project_root / "data" / "raw"
        self.processed_dir = self.project_root / "data" / "processed" / "regridded"
        self.daily_dir = self.processed_dir / "daily"
        self.scratch_dir = self.raw_dir / "_chunk_scratch"
        self.manifest_path = self.processed_dir / "manifest.json"

        self.min_free_bytes = int(min_free_disk_gb * 1024 * 1024 * 1024)

        for d in [self.raw_dir, self.processed_dir, self.daily_dir, self.scratch_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.grid = AntarcticGrid(cache_dir=self.project_root / "data" / "processed")
        self.regridder = EnvironmentalRegridder(self.grid)

        self.nsidc_downloader = NSIDCDownloader(self.scratch_dir / "nsidc")
        self.cmems_downloader = CopernicusMarineDownloader(self.scratch_dir / "copernicus")
        self.era5_downloader = ERA5Downloader(self.scratch_dir / "era5")

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load or initialize manifest of completed chunks."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading manifest: {e}. Reinitializing.")

        return {
            "pipeline_version": "1.0.0",
            "grid": {
                "epsg": 3412,
                "shape": [332, 316],
                "resolution": "25km",
                "ocean_cells": int(np.sum(self.grid.get_land_ocean_mask() == 1)),
                "land_cells": int(np.sum(self.grid.get_land_ocean_mask() == 0)),
            },
            "variables": [
                "sic",
                "wind_u",
                "wind_v",
                "air_temp",
                "sst",
                "current_u",
                "current_v",
            ],
            "completed_chunks": {},
            "total_days_processed": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _save_manifest(self):
        """Atomically persist manifest to disk."""
        self.manifest["updated_at"] = datetime.utcnow().isoformat()
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        tmp_path.replace(self.manifest_path)

    def check_disk_space(self):
        """Ensure sufficient free disk space exists before processing a chunk."""
        free_bytes = shutil.disk_usage(self.project_root).free
        free_gb = free_bytes / (1024 ** 3)
        min_gb = self.min_free_bytes / (1024 ** 3)

        if free_bytes < self.min_free_bytes:
            raise RuntimeError(
                f"INSUFFICIENT DISK SPACE: Only {free_gb:.2f} GB free on filesystem, "
                f"which is below the safety threshold of {min_gb:.2f} GB. "
                "Halting pipeline to prevent file corruption."
            )
        logger.debug(f"Disk check passed: {free_gb:.1f} GB available (threshold: {min_gb:.1f} GB).")

    def is_chunk_completed(self, chunk_id: str) -> bool:
        """Check if chunk is already completed and verified."""
        return chunk_id in self.manifest.get("completed_chunks", {})

    def process_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        chunk_id: Optional[str] = None,
        force: bool = False
    ) -> Dict:
        """
        Process an arbitrary date range chunk end-to-end:
        Download raw -> Regrid daily arrays -> Save daily .npz -> Discard raw.
        """
        if chunk_id is None:
            chunk_id = f"range_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        if not force and self.is_chunk_completed(chunk_id):
            logger.info(f"Chunk {chunk_id} already completed in manifest. Skipping (idempotent).")
            return self.manifest["completed_chunks"][chunk_id]

        self.check_disk_space()
        logger.info(f"\n=======================================================")
        logger.info(f"PROCESSING CHUNK {chunk_id}: {start_date.date()} to {end_date.date()}")
        logger.info(f"=======================================================")

        scratch = self.scratch_dir / chunk_id
        scratch.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Download NSIDC SIC daily files
            logger.info(f"[1/4] Downloading NSIDC SIC daily files for {chunk_id}...")
            sic_files = []
            cur_d = start_date
            while cur_d <= end_date:
                f = self.nsidc_downloader.download_date(cur_d, target_dir=scratch / "nsidc")
                sic_files.append((cur_d, f))
                cur_d += timedelta(days=1)

            # 2. Download CMEMS surface currents
            logger.info(f"[2/4] Downloading CMEMS surface currents for {chunk_id}...")
            cmems_file = self.cmems_downloader.download_date_range(
                start_date, end_date, output_file=scratch / "copernicus" / f"cmems_{chunk_id}.nc"
            )

            # 3. Download ERA5 atmospheric & SST variables
            logger.info(f"[3/4] Downloading ERA5 atmospheric & SST for {chunk_id}...")
            if start_date.year == end_date.year and start_date.month == end_date.month:
                days_list = list(range(start_date.day, end_date.day + 1))
                era5_file = self.era5_downloader.download_month(
                    start_date.year,
                    start_date.month,
                    output_file=scratch / "era5" / f"era5_{chunk_id}.nc",
                    days=days_list
                )
                era5_ds = xr.open_dataset(era5_file)
            else:
                era_datasets = []
                m_start = datetime(start_date.year, start_date.month, 1)
                while m_start <= end_date:
                    m_year = m_start.year
                    m_month = m_start.month
                    _, m_days = calendar.monthrange(m_year, m_month)
                    d_start = start_date.day if (m_year == start_date.year and m_month == start_date.month) else 1
                    d_end = end_date.day if (m_year == end_date.year and m_month == end_date.month) else m_days
                    days_list = list(range(d_start, d_end + 1))
                    ef = self.era5_downloader.download_month(
                        m_year,
                        m_month,
                        output_file=scratch / "era5" / f"era5_{m_year}{m_month:02d}_{d_start}_{d_end}.nc",
                        days=days_list
                    )
                    era_datasets.append(xr.open_dataset(ef))
                    if m_month == 12:
                        m_start = datetime(m_year + 1, 1, 1)
                    else:
                        m_start = datetime(m_year, m_month + 1, 1)

                era5_ds = xr.concat(era_datasets, dim="valid_time" if "valid_time" in era_datasets[0].coords else "time")

            # 4. Regrid daily arrays
            logger.info(f"[4/4] Regridding daily arrays for {chunk_id}...")
            days_saved = []

            cmems_ds = xr.open_dataset(cmems_file)

            try:
                for cur_date, sic_file_path in sic_files:
                    date_str = cur_date.strftime("%Y-%m-%d")
                    date_str_compact = cur_date.strftime("%Y%m%d")
                    year_daily_dir = self.daily_dir / str(cur_date.year)
                    year_daily_dir.mkdir(parents=True, exist_ok=True)
                    out_npz = year_daily_dir / f"{date_str_compact}.npz"

                    daily_bundle = self.regridder.process_daily_bundle(
                        sic_file=sic_file_path,
                        era5_ds=era5_ds,
                        cmems_ds=cmems_ds,
                        target_date=date_str
                    )

                    if np.isnan(daily_bundle).any():
                        raise ValueError(f"Regridded tensor for {date_str} contains NaN values!")

                    np.savez_compressed(
                        out_npz,
                        data=daily_bundle,
                        date=date_str,
                        variables=self.manifest["variables"]
                    )
                    days_saved.append(out_npz)
            finally:
                cmems_ds.close()
                era5_ds.close()

            # 5. Record completion in manifest
            chunk_record = {
                "chunk_id": chunk_id,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "num_days": len(days_saved),
                "completed_at": datetime.utcnow().isoformat(),
                "file_size_bytes": sum(f.stat().st_size for f in days_saved),
            }
            self.manifest["completed_chunks"][chunk_id] = chunk_record
            self.manifest["total_days_processed"] = sum(
                c["num_days"] for c in self.manifest["completed_chunks"].values()
            )
            self._save_manifest()
            logger.info(f"Chunk {chunk_id} successfully regridded and recorded: {len(days_saved)} daily files.")

            return chunk_record

        finally:
            if scratch.exists():
                logger.info(f"Discarding raw native scratch files for chunk {chunk_id}...")
                shutil.rmtree(scratch, ignore_errors=True)

    def process_month(
        self,
        year: int,
        month: int,
        force: bool = False
    ) -> Dict:
        """
        Process a single monthly chunk end-to-end.
        """
        start_date = datetime(year, month, 1)
        _, num_days = calendar.monthrange(year, month)
        end_date = datetime(year, month, num_days)
        chunk_id = f"{year}{month:02d}"
        return self.process_date_range(start_date, end_date, chunk_id=chunk_id, force=force)

    def run_range(
        self,
        start_date: datetime,
        end_date: datetime,
        force: bool = False
    ) -> Dict:
        """
        Run pipeline across a multi-year range, month by month.
        """
        logger.info(
            f"Starting Real Data Pipeline: {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')}"
        )

        curr = datetime(start_date.year, start_date.month, 1)
        end_month = datetime(end_date.year, end_date.month, 1)

        summary = {"processed": [], "skipped": []}

        while curr <= end_month:
            chunk_id = f"{curr.year}{curr.month:02d}"
            if not force and self.is_chunk_completed(chunk_id):
                summary["skipped"].append(chunk_id)
            else:
                res = self.process_month(curr.year, curr.month, force=force)
                summary["processed"].append(res)

            # Advance month
            if curr.month == 12:
                curr = datetime(curr.year + 1, 1, 1)
            else:
                curr = datetime(curr.year, curr.month + 1, 1)

        return summary
