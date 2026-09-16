import sys
import os
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append("/Users/pratiksmac/Downloads/icccy/iceberg-drift/src")

from iceberg_drift.data_processing.download import (
    _load_byu_local_icebergs,
    _download_nic_icebergs,
    get_chunked_bboxes,
    download_era5_wind,
    download_copernicus_currents
)

def main():
    start_date = "2015-01-01"
    end_date = "2022-12-31"
    
    print(f"Loading real iceberg data from {start_date} to {end_date} to compute bounding boxes...")
    byu_df = _load_byu_local_icebergs(
        data_dir="/Users/pratiksmac/Downloads/data/raw/iceberg_positions",
        start_date=start_date,
        end_date=end_date,
        min_length_m=100
    )
    nic_df = _download_nic_icebergs(
        start_date=start_date,
        end_date=end_date,
        min_length_m=100,
        local_path="/Users/pratiksmac/Downloads/data/raw/iceberg_positions/iceberg_positions"
    )
    df = pd.concat([byu_df, nic_df], ignore_index=True)
    df = df.dropna(subset=['lat', 'lon'])
    
    if len(df) == 0:
        print("No valid iceberg data found for this date range.")
        return
        
    bboxes = get_chunked_bboxes(df, buffer=2.0)
    print(f"Computed {len(bboxes)} bounding boxes for environmental data based on real iceberg trajectories.")
    
    raw_dir = Path("/Users/pratiksmac/Downloads/data/raw")
    era5_dir = raw_dir / "era5"
    cop_dir = raw_dir / "copernicus"
    
    era5_dir.mkdir(parents=True, exist_ok=True)
    cop_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Downloading Copernicus Currents ---")
    try:
        download_copernicus_currents(
            output_dir=str(cop_dir),
            start_date=start_date,
            end_date=end_date,
            bboxes=[bboxes[0]],
            depth_levels=[0.49402499198913574],
            allow_synthetic_fallback=False
        )
        print("Copernicus download completed successfully!")
    except Exception as e:
        print(f"Failed Copernicus download: {e}")
        
    print("\n--- Downloading ERA5 Wind ---")
    try:
        download_era5_wind(
            output_dir=str(era5_dir),
            start_date=start_date,
            end_date=end_date,
            bboxes=[bboxes[0]],
            allow_synthetic_fallback=False
        )
        print("ERA5 download completed successfully!")
    except Exception as e:
        print(f"Failed ERA5 download: {e}")

if __name__ == "__main__":
    main()
