#!/bin/bash
# Monitor the download progress for both NSIDC SIC and ERA5 data

echo "=========================================="
echo "DATA DOWNLOAD PROGRESS MONITOR"
echo "=========================================="
echo ""

# Check NSIDC SIC progress
echo "--- NSIDC SEA-ICE CONCENTRATION ---"
echo "Target: ~4,018 files (2008-2018, 11 years)"
echo ""
for year in {2008..2018}; do
    count=$(find data/data/sic/$year -name "*.nc" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$year" -eq 2008 ]; then
        expected=366
    elif [ "$year" -eq 2012 ] || [ "$year" -eq 2016 ]; then
        expected=366
    else
        expected=365
    fi
    printf "%d: %3d/%d files" $year $count $expected
    if [ $count -eq $expected ]; then
        echo " ✓ COMPLETE"
    elif [ $count -gt 0 ]; then
        pct=$((count * 100 / expected))
        echo " (${pct}% done)"
    else
        echo " (not started)"
    fi
done
total_sic=$(find data/data/sic -name "*.nc" 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Total SIC files: $total_sic / ~4,018"
echo ""

# Check ERA5 progress
echo "--- ERA5 ATMOSPHERIC FORCING ---"
echo "Target: 132 monthly files (2008-2018)"
echo ""
for year in {2008..2018}; do
    count=$(find data/data/raw/era5/$year -name "era5_forcing_*.nc" 2>/dev/null | wc -l | tr -d ' ')
    printf "%d: %2d/12 months" $year $count
    if [ $count -eq 12 ]; then
        echo " ✓ COMPLETE"
    elif [ $count -gt 0 ]; then
        echo " (downloading...)"
    else
        echo " (not started)"
    fi
done
total_era5=$(find data/data/raw/era5 -name "era5_forcing_*.nc" 2>/dev/null | wc -l | tr -d ' ')
era5_size=$(du -sh data/data/raw/era5 2>/dev/null | awk '{print $1}')
echo ""
echo "Total ERA5 files: $total_era5 / 132"
echo "Total ERA5 size: ${era5_size:-0B}"
echo ""

# Check for missing dates report
if [ -f data/data/sic/MISSING_DATES.json ]; then
    missing_count=$(cat data/data/sic/MISSING_DATES.json | grep -c '"date"')
    echo "--- MISSING DATES REPORT ---"
    echo "Found MISSING_DATES.json with $missing_count missing days"
    echo "This is normal for sensor gaps/outages in 2008-2018"
    echo ""
fi

# Check disk space
echo "--- DISK SPACE ---"
df -h . | tail -1 | awk '{print "Available: " $4}'
echo ""

# Check if processes are running
echo "--- PROCESS STATUS ---"
if ps aux | grep -q "[d]ownload_nsidc_sic.py"; then
    echo "✓ NSIDC download process is RUNNING"
else
    echo "✗ NSIDC download process is NOT running"
fi

if ps aux | grep -q "[d]ownload_era5_forcing.py"; then
    echo "✓ ERA5 download process is RUNNING"
else
    echo "✗ ERA5 download process is NOT running"
fi
echo ""
echo "=========================================="
