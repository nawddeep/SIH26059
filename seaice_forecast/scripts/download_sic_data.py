#!/usr/bin/env python3
"""
Download NSIDC Sea-Ice Concentration data for Antarctic region.

This script downloads daily sea-ice concentration data from NSIDC,
which is CRITICAL for training the sea-ice forecasting models.

Requirements:
    pip install earthaccess
    
Usage:
    python scripts/download_sic_data.py --start-year 2008 --end-year 2026
    
    # Or download specific year:
    python scripts/download_sic_data.py --start-year 2015 --end-year 2015
    
First-time setup:
    1. Register free account at: https://urs.earthdata.nasa.gov/users/new
    2. Run this script - it will prompt for username/password
    3. Credentials are cached for future runs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False
    print("ERROR: earthaccess not installed")
    print("Install with: pip install earthaccess")
    sys.exit(1)


def setup_earthdata_auth():
    """
    Setup NASA Earthdata authentication.
    You'll be prompted for username/password on first run.
    
    Register free account at: https://urs.earthdata.nasa.gov/users/new
    """
    print("\n" + "="*70)
    print("NASA EARTHDATA AUTHENTICATION")
    print("="*70)
    print()
    print("NSIDC data requires a free NASA Earthdata account.")
    print()
    print("If you don't have an account:")
    print("  1. Go to: https://urs.earthdata.nasa.gov/users/new")
    print("  2. Register (takes 2 minutes)")
    print("  3. Verify your email")
    print("  4. Come back and run this script again")
    print()
    print("If you already have an account, enter credentials below:")
    print("(They will be cached securely for future runs)")
    print()
    
    try:
        auth = earthaccess.login(strategy="interactive")
        
        if auth.authenticated:
            print()
            print("✓ Authentication successful!")
            print()
            return True
        else:
            print()
            print("✗ Authentication failed.")
            print("  Please check your username and password.")
            print("  Reset password at: https://urs.earthdata.nasa.gov/")
            print()
            return False
            
    except Exception as e:
        print(f"\n✗ Authentication error: {e}")
        return False


def download_sic_year(year, output_dir, hemisphere='south'):
    """
    Download daily SIC data for one year.
    
    Args:
        year: Year to download (e.g., 2015)
        output_dir: Output directory path
        hemisphere: 'south' for Antarctic, 'north' for Arctic
    """
    output_path = Path(output_dir) / str(year)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Downloading SIC data for {year} ({hemisphere}ern hemisphere)")
    print(f"{'='*70}")
    
    # Determine bounding box
    if hemisphere == 'south':
        bbox = (-180, -90, 180, -55)  # Antarctic
        print(f"  Region: Antarctic (-90° to -55° latitude)")
    else:
        bbox = (-180, 55, 180, 90)    # Arctic
        print(f"  Region: Arctic (55° to 90° latitude)")
    
    print(f"  Output: {output_path}")
    print()
    
    try:
        # Try multiple product IDs (NSIDC uses different codes)
        product_ids = [
            ('NSIDC-0051', 'Near-Real-Time product (NSIDC-0051)'),
            ('NSIDC-0081', 'Climate Data Record v4 (NSIDC-0081)'),
            ('G02202', 'Climate Data Record (G02202)'),
            ('SPL2SMP', 'SMAP Enhanced L2 (alternative)'),
        ]
        
        results = []
        for product_id, description in product_ids:
            if results:
                break  # Found data, stop searching
                
            print(f"  Searching for {description}...")
            try:
                results = earthaccess.search_data(
                    short_name=product_id,
                    temporal=(f"{year}-01-01", f"{year}-12-31"),
                    bounding_box=bbox
                )
                
                if results:
                    print(f"  ✓ Found data using {product_id}")
                    break
            except Exception as e:
                print(f"  ✗ {product_id} failed: {str(e)[:50]}")
                continue
        
        if not results:
            print(f"  ✗ No data found for {year} using any product ID")
            print(f"     Tried: NSIDC-0051, NSIDC-0081, G02202")
            print()
            print(f"  Alternative: Manual download")
            print(f"    1. Go to: https://nsidc.org/data/nsidc-0051")
            print(f"    2. Select: Southern Hemisphere, {year}")
            print(f"    3. Download files to: {output_path}")
            return False
        
        print(f"  ✓ Found {len(results)} granules")
        
        # Show first few file names for verification
        if len(results) > 0:
            print(f"  Sample file: {results[0]['meta']['native-id']}")
        
        print(f"  Downloading to {output_path}...")
        print(f"  (This may take 5-15 minutes...)")
        
        # Download files
        downloaded = earthaccess.download(
            results,
            str(output_path)
        )
        
        if downloaded:
            # Calculate total size
            total_size = sum(Path(f).stat().st_size for f in downloaded if Path(f).exists())
            size_mb = total_size / (1024**2)
            
            print()
            print(f"  ✓ Successfully downloaded {len(downloaded)} files")
            print(f"  ✓ Total size: {size_mb:.1f} MB")
            print(f"  ✓ Location: {output_path}")
            return True
        else:
            print(f"  ✗ Download failed - no files retrieved")
            return False
        
    except Exception as e:
        print(f"  ✗ Error downloading {year}: {e}")
        print(f"     Error type: {type(e).__name__}")
        return False


def verify_downloads(output_dir, start_year, end_year):
    """Verify downloaded data and print summary."""
    print("\n" + "="*70)
    print("DOWNLOAD VERIFICATION")
    print("="*70)
    print()
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print("✗ Output directory does not exist")
        return
    
    years_found = []
    years_missing = []
    total_files = 0
    total_size_mb = 0
    
    for year in range(start_year, end_year + 1):
        year_path = output_path / str(year)
        
        if year_path.exists():
            nc_files = list(year_path.glob("*.nc"))
            
            if nc_files:
                year_size = sum(f.stat().st_size for f in nc_files)
                size_mb = year_size / (1024**2)
                
                years_found.append(year)
                total_files += len(nc_files)
                total_size_mb += size_mb
                
                status = "✓"
                if len(nc_files) < 365:
                    status = "⚠"
                
                print(f"  {status} {year}: {len(nc_files):3d} files ({size_mb:6.1f} MB)")
            else:
                years_missing.append(year)
                print(f"  ✗ {year}: No files")
        else:
            years_missing.append(year)
            print(f"  ✗ {year}: Directory not found")
    
    print()
    print("-" * 70)
    print(f"Summary:")
    print(f"  Years found:     {len(years_found)}/{end_year - start_year + 1}")
    print(f"  Total files:     {total_files}")
    print(f"  Total size:      {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    if years_missing:
        print(f"  Missing years:   {', '.join(map(str, years_missing))}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Download NSIDC Sea-Ice Concentration data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all years (2008-2026)
  python scripts/download_sic_data.py
  
  # Download specific year
  python scripts/download_sic_data.py --start-year 2015 --end-year 2015
  
  # Download recent years only
  python scripts/download_sic_data.py --start-year 2020 --end-year 2026
  
  # Download Arctic data instead of Antarctic
  python scripts/download_sic_data.py --hemisphere north

First time setup:
  1. Register at: https://urs.earthdata.nasa.gov/users/new
  2. Run this script
  3. Enter your credentials when prompted
        """
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
    parser.add_argument(
        "--hemisphere",
        choices=['south', 'north'],
        default='south',
        help="Hemisphere: south (Antarctic) or north (Arctic) [default: south]"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads, don't download new data"
    )
    
    args = parser.parse_args()
    
    print()
    print("="*70)
    print("NSIDC SEA-ICE CONCENTRATION DATA DOWNLOADER")
    print("="*70)
    print()
    print(f"Date range:  {args.start_year} - {args.end_year}")
    print(f"Hemisphere:  {args.hemisphere}ern ({'Antarctic' if args.hemisphere == 'south' else 'Arctic'})")
    print(f"Output dir:  {args.output_dir}")
    print()
    
    # Verify-only mode
    if args.verify_only:
        verify_downloads(args.output_dir, args.start_year, args.end_year)
        return
    
    # Check earthaccess installation
    if not EARTHACCESS_AVAILABLE:
        print("✗ ERROR: earthaccess package not installed")
        print()
        print("Install with:")
        print("  pip install earthaccess")
        print()
        sys.exit(1)
    
    # Authenticate
    if not setup_earthdata_auth():
        print("\n✗ Authentication failed. Cannot proceed.")
        print()
        print("Troubleshooting:")
        print("  1. Check username/password are correct")
        print("  2. Verify email is confirmed")
        print("  3. Try resetting password")
        print("  4. Clear cached credentials: rm -rf ~/.netrc ~/.urs_cookies")
        print()
        sys.exit(1)
    
    # Download year by year
    success_years = []
    failed_years = []
    
    for year in range(args.start_year, args.end_year + 1):
        if download_sic_year(year, args.output_dir, args.hemisphere):
            success_years.append(year)
        else:
            failed_years.append(year)
    
    # Final verification
    verify_downloads(args.output_dir, args.start_year, args.end_year)
    
    # Summary
    print()
    print("="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print()
    print(f"✓ Successful:  {len(success_years)} years")
    
    if success_years:
        print(f"   {', '.join(map(str, success_years))}")
    
    if failed_years:
        print()
        print(f"✗ Failed:      {len(failed_years)} years")
        print(f"   {', '.join(map(str, failed_years))}")
        print()
        print("   These years may:")
        print("   - Not be available yet (future dates)")
        print("   - Use different product IDs")
        print("   - Have processing delays")
        print("   - Try downloading manually from NSIDC website")
    
    print()
    print("="*70)
    
    if len(success_years) > 0:
        print("✓ DOWNLOAD COMPLETE")
        print()
        print("Next steps:")
        print()
        print("1. Verify files are correct format:")
        print(f"   ls -lh {args.output_dir}/2015/")
        print()
        print("2. Run data preparation:")
        print("   python scripts/prepare_data.py")
        print()
        print("3. Start training:")
        print("   python scripts/training/train_phase1.py --epochs 50")
        print()
    else:
        print("✗ NO DATA DOWNLOADED")
        print()
        print("Check authentication and internet connection.")
        print()
    
    print("="*70)
    
    sys.exit(0 if len(failed_years) == 0 else 1)


if __name__ == "__main__":
    main()
