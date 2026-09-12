#!/usr/bin/env python
"""
Download NSIDC Sea Ice Concentration CDR v6 data.

Usage:
    python scripts/download_data.py --start-date 2010-01-01 --end-date 2022-12-31
    python scripts/download_data.py --start-date 2010-01-01 --end-date 2010-01-31 --check-only
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seaice_forecast.config import load_config, resolve_paths
from seaice_forecast.data_processing.download import NSIDCDownloader
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download NSIDC Sea Ice Concentration CDR v6 data"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: from config)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check data availability, don't download"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: use default config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config()
    
    config = resolve_paths(config)
    
    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config['data']['paths']['raw']
    
    logger.info(f"Output directory: {output_dir}")
    
    # Create downloader
    downloader = NSIDCDownloader(
        output_dir=output_dir,
        hemisphere=config['data']['nsidc']['hemisphere']
    )
    
    if args.check_only:
        # Check availability
        logger.info("Checking data availability (sampling first 100 dates)...")
        stats = downloader.check_data_availability(args.start_date, args.end_date)
        
        print("\n" + "="*60)
        print("DATA AVAILABILITY CHECK")
        print("="*60)
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Checked: {stats['total_checked']} dates")
        print(f"Available: {stats['available']}")
        print(f"Missing: {stats['missing']}")
        print(f"Availability rate: {stats['availability_rate']:.1%}")
        print("="*60)
        
    else:
        # Download data
        logger.info(f"Downloading data from {args.start_date} to {args.end_date}...")
        files = downloader.download_date_range(args.start_date, args.end_date)
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Downloaded: {len(files)} files")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        if len(files) > 0:
            print("\nNext step: Run preparation script")
            print("  python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
