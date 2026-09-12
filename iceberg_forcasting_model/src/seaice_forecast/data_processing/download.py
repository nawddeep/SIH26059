"""
Download NOAA/NSIDC Sea Ice Concentration CDR v6 data.

NSIDC CDR v6 (G02202) provides daily sea-ice concentration on polar stereographic grids.
Antarctic (South) data is available at 25km resolution (316x332 grid).

Data URL pattern:
https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/YYYY/seaice_conc_daily_sh_YYYYMMDD_f18_v06.nc
"""

import os
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NSIDCDownloader:
    """Download NSIDC Sea Ice Concentration CDR v6 data."""
    
    def __init__(self, output_dir: str, hemisphere: str = "south"):
        """
        Initialize NSIDC downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            hemisphere: 'south' for Antarctic or 'north' for Arctic
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hemisphere = hemisphere
        
        # Base URL for NSIDC CDR v6
        self.base_url = f"https://noaadata.apps.nsidc.org/NOAA/G02202_V6/{hemisphere}/daily"
        
        # Satellite sensor codes (ordered by priority - newer preferred)
        # f18 = SSMIS on DMSP F18 (2010-present)
        # f17 = SSMIS on DMSP F17 (2006-2023)
        self.sensors = ["f18", "f17"]
    
    def get_filename(self, date: datetime, sensor: str) -> str:
        """
        Get NSIDC filename for a specific date and sensor.
        
        Args:
            date: Date to download
            sensor: Sensor code (e.g., 'f18')
            
        Returns:
            Filename string
        """
        date_str = date.strftime("%Y%m%d")
        hemi_code = "sh" if self.hemisphere == "south" else "nh"
        return f"seaice_conc_daily_{hemi_code}_{date_str}_{sensor}_v06r01.nc"
    
    def get_url(self, date: datetime, sensor: str) -> str:
        """
        Get download URL for a specific date and sensor.
        
        Args:
            date: Date to download
            sensor: Sensor code
            
        Returns:
            Full URL string
        """
        year = date.year
        filename = self.get_filename(date, sensor)
        return f"{self.base_url}/{year}/{filename}"
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download a single file.
        
        Args:
            url: URL to download from
            output_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Write file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Failed to download {url}: {e}")
            return False
    
    def download_date(self, date: datetime) -> Optional[Path]:
        """
        Download data for a specific date, trying multiple sensors.
        
        Args:
            date: Date to download
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        for sensor in self.sensors:
            url = self.get_url(date, sensor)
            filename = self.get_filename(date, sensor)
            output_path = self.output_dir / filename
            
            # Skip if already exists
            if output_path.exists():
                logger.debug(f"File already exists: {filename}")
                return output_path
            
            # Try to download
            if self.download_file(url, output_path):
                logger.info(f"Downloaded: {filename}")
                return output_path
        
        logger.warning(f"No data available for {date.strftime('%Y-%m-%d')}")
        return None
    
    def download_date_range(
        self, 
        start_date: str, 
        end_date: str,
        skip_existing: bool = True
    ) -> List[Path]:
        """
        Download data for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            skip_existing: Skip files that already exist
            
        Returns:
            List of paths to downloaded files
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate all dates in range
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        
        logger.info(f"Downloading {len(dates)} days of data from {start_date} to {end_date}")
        
        downloaded_files = []
        failed_dates = []
        
        for date in tqdm(dates, desc="Downloading"):
            file_path = self.download_date(date)
            if file_path:
                downloaded_files.append(file_path)
            else:
                failed_dates.append(date)
        
        logger.info(f"Successfully downloaded: {len(downloaded_files)} files")
        if failed_dates:
            logger.warning(f"Failed to download {len(failed_dates)} dates")
            logger.debug(f"Failed dates: {[d.strftime('%Y-%m-%d') for d in failed_dates[:10]]}")
        
        return downloaded_files
    
    def check_data_availability(
        self, 
        start_date: str, 
        end_date: str
    ) -> dict:
        """
        Check which dates have available data without downloading.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with availability statistics
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        
        available = 0
        missing = 0
        
        logger.info(f"Checking availability for {len(dates)} dates...")
        
        for date in tqdm(dates[:100], desc="Checking"):  # Sample first 100
            for sensor in self.sensors:
                url = self.get_url(date, sensor)
                try:
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        available += 1
                        break
                except:
                    continue
            else:
                missing += 1
        
        return {
            "total_checked": len(dates[:100]),
            "available": available,
            "missing": missing,
            "availability_rate": available / len(dates[:100]) if dates else 0
        }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download NSIDC SIC data")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/raw/nsidc", help="Output directory")
    parser.add_argument("--check-only", action="store_true", help="Only check availability")
    
    args = parser.parse_args()
    
    downloader = NSIDCDownloader(args.output_dir)
    
    if args.check_only:
        stats = downloader.check_data_availability(args.start_date, args.end_date)
        print(f"\nData Availability Check:")
        print(f"  Total checked: {stats['total_checked']}")
        print(f"  Available: {stats['available']}")
        print(f"  Missing: {stats['missing']}")
        print(f"  Rate: {stats['availability_rate']:.1%}")
    else:
        files = downloader.download_date_range(args.start_date, args.end_date)
        print(f"\nDownloaded {len(files)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
