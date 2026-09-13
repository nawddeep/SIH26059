#!/usr/bin/env python3
"""
CLI Runner for the Real Polar Data Ingestion & Regridding Pipeline.

Usage:
  # End-to-end dry run (7 days across the 2021-06-30 CMEMS seam):
  python scripts/data/run_data_pipeline.py --dry-run

  # Run full year:
  python scripts/data/run_data_pipeline.py --start-date 2021-01-01 --end-date 2021-12-31

  # Force re-processing:
  python scripts/data/run_data_pipeline.py --start-date 2021-06-01 --end-date 2021-07-31 --force
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
src_dir = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_dir))

from seaice_forecast.data_processing.pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_data_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run production real data pipeline for Antarctic sea-ice forecasting."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run short dry-run range across the CMEMS 2021-06-30 seam."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-processing of completed chunks."
    )
    parser.add_argument(
        "--min-disk-gb",
        type=float,
        default=5.0,
        help="Minimum required free disk space in GB (default: 5.0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = DataPipeline(min_free_disk_gb=args.min_disk_gb)

    if args.dry_run:
        logger.info("Executing dry-run mode: testing chunked execution across the 2021-06-30 CMEMS seam...")
        # Test 2021-06-28 to 2021-07-04 (7 days across seam)
        start_date = datetime(2021, 6, 28)
        end_date = datetime(2021, 7, 4)
        res = pipeline.process_date_range(start_date, end_date, chunk_id="dryrun_seam_test", force=args.force)
        summary = {"processed": [res], "skipped": []}
    else:
        if not args.start_date or not args.end_date:
            logger.error("Must specify --start-date and --end-date, or use --dry-run.")
            sys.exit(1)
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        summary = pipeline.run_range(start_date, end_date, force=args.force)

    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Processed chunks: {len(summary['processed'])}")
    logger.info(f"Skipped (already complete): {len(summary['skipped'])}")
    logger.info(f"Total days processed to date: {pipeline.manifest.get('total_days_processed', 0)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
