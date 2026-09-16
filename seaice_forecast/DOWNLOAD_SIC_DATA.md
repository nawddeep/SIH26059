# Download Sea-Ice Concentration (SIC) Data - CRITICAL STEP

## What You Need to Download

**Dataset**: NSIDC Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS Passive Microwave Data, Version 4

**Official Name**: NSIDC-0051 (or newer G02202 Climate Data Record)

**Coverage Needed**: 
- **Region**: Southern Hemisphere (Antarctic)
- **Years**: 2008-2026
- **Temporal Resolution**: Daily
- **Spatial Resolution**: 25km polar stereographic grid

---

## Option 1: Automatic Download (Python Script)

Save this as `scripts/download_sic_data.py`:

```python
#!/usr/bin/env python3
"""
Download NSIDC Sea-Ice Concentration data for Antarctic region.

Requirements:
    pip install earthaccess requests
    
Usage:
    python scripts/download_sic_data.py --start-year 2008 --end-year 2026
"""

import argparse
import earthaccess
from pathlib import Path
from datetime import datetime, timedelta
import sys

def setup_earthdata_auth():
    """
    Setup NASA Earthdata authentication.
    You'll be prompted for username/password on first run.
    
    Register free account at: https://urs.earthdata.nasa.gov/users/new
    """
    print("Setting up NASA Earthdata authentication...")
    print("If you don't have an account, register at:")
    print("https://urs.earthdata.nasa.gov/users/new")
    print()
    
    auth = earthaccess.login(strategy="interactive")
    
    if auth.authenticated:
        print("✓ Authentication successful!")
        return True
    else:
        print("✗ Authentication failed. Please check credentials.")
        return False


def download_sic_daily(year, output_dir):
    """
    Download daily SIC data for one year.
    
    Args:
        year: Year to download (e.g., 2015)
        output_dir: Output directory path
    """
    output_path = Path(output_dir) / str(year)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading SIC data for {year}")
    print(f"{'='*70}")
    
    # Search for NSIDC-0051 or G02202 data
    # Short name varies: NSIDC-0051 (older) or NSIDC-0081 (CDR v4)
    try:
        # Try Climate Data Record first (newer, better)
        results = earthaccess.search_data(
            short_name='NSIDC-0081',  # Sea Ice Concentrations CDR v4
            temporal=(f"{year}-01-01", f"{year}-12-31"),
            bounding_box=(-180, -90, 180, -60)  # Southern Hemisphere
        )
        
        if not results:
            # Fallback to older product
            print("  CDR not found, trying legacy product...")
            results = earthaccess.search_data(
                short_name='NSIDC-0051',
                temporal=(f"{year}-01-01", f"{year}-12-31"),
                bounding_box=(-180, -90, 180, -60)
            )
        
        if not results:
            print(f"  ✗ No data found for {year}")
            return False
        
        print(f"  Found {len(results)} files")
        
        # Download files
        downloaded = earthaccess.download(
            results,
            str(output_path)
        )
        
        print(f"  ✓ Downloaded {len(downloaded)} files to {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading {year}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download NSIDC Sea-Ice Concentration data"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2008,
        help="Start year (default: 2008)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="End year (default: 2026)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/data/sic",
        help="Output directory (default: data/data/sic)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("NSIDC SEA-ICE CONCENTRATION DATA DOWNLOADER")
    print("="*70)
    print()
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Authenticate
    if not setup_earthdata_auth():
        print("\nAuthentication failed. Cannot proceed.")
        sys.exit(1)
    
    # Download year by year
    success_count = 0
    fail_count = 0
    
    for year in range(args.start_year, args.end_year + 1):
        if download_sic_daily(year, args.output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print()
    print("="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✓ Successful: {success_count} years")
    print(f"✗ Failed: {fail_count} years")
    print()
    
    if success_count > 0:
        print("Next steps:")
        print("  1. Verify downloaded files:")
        print(f"     ls -lh {args.output_dir}/*/")
        print("  2. Run data preparation:")
        print("     python scripts/prepare_data.py")
    
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
```

### Run the script:

```bash
# Install required package
pip install earthaccess

# Run download
cd ~/Documents/iceberg_models/iceberg_models-main/seaice_forecast
python scripts/download_sic_data.py --start-year 2008 --end-year 2026
```

**You'll be prompted for:**
- NASA Earthdata username
- NASA Earthdata password

**Don't have an account?** Register free at: https://urs.earthdata.nasa.gov/users/new

---

## Option 2: Manual Download (Web Interface)

### Step 1: Create NASA Earthdata Account
1. Go to: https://urs.earthdata.nasa.gov/users/new
2. Fill in registration form
3. Verify email
4. Login

### Step 2: Access NSIDC Data Portal
1. Go to: https://nsidc.org/data/nsidc-0051
2. Click "Access Data" button
3. Login with Earthdata credentials

### Step 3: Download Settings
**Data Selection:**
- **Hemisphere**: Southern (Antarctic)
- **Date Range**: 2008-01-01 to 2026-12-31
- **File Format**: NetCDF (preferred) or GeoTIFF
- **Temporal Resolution**: Daily
- **Spatial Resolution**: 25km

**Recommended Subset:**
- **Latitude**: -90° to -55° (Antarctic region)
- **Variables**: Sea ice concentration (primary)

### Step 4: Download Files
**Batch download options:**
1. **wget script** (recommended for large downloads)
   - NSIDC provides a wget script after selection
   - Run: `bash download_script.sh`

2. **Direct download** (small date ranges)
   - Click individual files
   - Save to: `data/data/sic/YYYY/` (create folders per year)

---

## Option 3: Use NSIDC Data Access Tool

### Install nsidc-data-access:
```bash
pip install nsidc-data-access
```

### Download command:
```bash
nsidc-download \
    --dataset NSIDC-0051 \
    --start-date 2008-01-01 \
    --end-date 2026-12-31 \
    --hemisphere south \
    --output-dir data/data/sic
```

---

## Expected File Structure After Download

```
data/data/sic/
├── 2008/
│   ├── seaice_conc_daily_sh_20080101_v04r00.nc
│   ├── seaice_conc_daily_sh_20080102_v04r00.nc
│   └── ... (365 files)
├── 2009/
│   └── ... (365 files)
...
├── 2026/
│   └── ... (365 files)
```

**Expected size**: 
- ~50-100 MB per year
- ~1-2 GB total for 2008-2026

---

## Verify Downloaded Data

```bash
cd ~/Documents/iceberg_models/iceberg_models-main/seaice_forecast

python3 << 'EOF'
from pathlib import Path
import os

sic_dir = Path("data/data/sic")

if not sic_dir.exists():
    print("✗ SIC directory not found")
else:
    years = sorted([d for d in sic_dir.iterdir() if d.is_dir()])
    
    print("SIC Data Verification:")
    print("=" * 50)
    
    for year_dir in years:
        nc_files = list(year_dir.glob("*.nc"))
        year = year_dir.name
        
        if nc_files:
            total_size = sum(f.stat().st_size for f in nc_files)
            size_mb = total_size / (1024**2)
            print(f"  {year}: {len(nc_files)} files ({size_mb:.1f} MB)")
        else:
            print(f"  {year}: NO FILES")
    
    print("=" * 50)
    print(f"\nTotal years: {len(years)}")
EOF
```

---

## Alternative Sources (If NSIDC Access Issues)

### 1. Copernicus Marine Service (EU)
```
Dataset: SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_001
URL: https://marine.copernicus.eu/
Coverage: 2007-present
Resolution: 25km
```

### 2. OSI SAF (EUMETSAT)
```
Dataset: Global Sea Ice Concentration
URL: https://osi-saf.eumetsat.int/
Product: OSI-401-d (daily)
Coverage: 1978-present
```

### 3. NOAA/NCEI
```
Dataset: Sea Ice Index
URL: https://nsidc.org/data/seaice_index
Format: Binary or NetCDF
Coverage: 1978-present
```

---

## Troubleshooting

### Issue: Authentication Failed
**Solution**: 
```bash
# Clear cached credentials
rm -rf ~/.netrc ~/.urs_cookies

# Try again
python scripts/download_sic_data.py
```

### Issue: Download Very Slow
**Solution**: Use parallel download or download smaller date ranges

### Issue: "Data not available for this date"
**Solution**: 
- Check data availability calendar on NSIDC
- Some recent dates may have processing delays
- Use latest available date instead of 2026

### Issue: Files are in wrong format
**Solution**: 
- Request NetCDF format (not GeoTIFF or binary)
- Or convert after download using `gdal_translate`

---

## After Download Complete

**Run data preparation:**
```bash
cd ~/Documents/iceberg_models/iceberg_models-main/seaice_forecast
python scripts/prepare_data.py
```

This will:
1. Read SIC NetCDF files
2. Regrid to Antarctic polar stereographic (25km)
3. Match with wind/current data
4. Create train/val/test splits
5. Save processed `train_data.npz` and `val_data.npz`

**Then you can train:**
```bash
python scripts/training/train_phase1.py --epochs 50
```

---

## Quick Start Command (Copy-Paste)

```bash
# 1. Install dependencies
pip install earthaccess

# 2. Download SIC data (will prompt for NASA Earthdata login)
cd ~/Documents/iceberg_models/iceberg_models-main/seaice_forecast
python scripts/download_sic_data.py --start-year 2008 --end-year 2026

# 3. Verify download
ls -lh data/data/sic/*/

# 4. Prepare data
python scripts/prepare_data.py

# 5. Start training
python scripts/training/train_phase1.py --epochs 50
```

---

## Important Notes

1. **NASA Earthdata account is FREE** - no payment required
2. **Download time**: 1-3 hours depending on connection speed
3. **Storage needed**: ~2 GB for full dataset (2008-2026)
4. **Data usage policy**: NSIDC data is free for research/education use
5. **Citation required**: When publishing results, cite NSIDC-0051 dataset

---

## Data Citation

```
Cavalieri, D. J., C. L. Parkinson, P. Gloersen, and H. J. Zwally. 1996, 
updated yearly. Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS 
Passive Microwave Data, Version 1. Boulder, Colorado USA. 
NASA National Snow and Ice Data Center Distributed Active Archive Center. 
https://doi.org/10.5067/8GQ8LZQVL0VL
```

---

## Help & Support

**NSIDC Support**: nsidc@nsidc.org  
**Data User Guide**: https://nsidc.org/sites/default/files/nsidc-0051-v001-userguide.pdf  
**FAQ**: https://nsidc.org/support

**For this project**: Check issues in project repository or contact maintainer.
