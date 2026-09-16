# Antarctic Sea-Ice Training Data Download - Status Report

**Started:** September 14, 2026, 21:00  
**Repository:** `/Users/nawdddep/Documents/iceberg_models/iceberg_models-main/seaice_forecast`

## Executive Summary

Two long-running download processes have been started to fetch the missing training data for the Antarctic sea-ice forecasting model. Both processes are running in the background and are fully resumable.

### Current Status (as of 21:15)

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| **NSIDC SIC** | ✓ Running | 508/4,018 files (13%) | 2008 complete, 2009 in progress |
| **ERA5 Forcing** | ✓ Running | 1/132 months (1%) | 2008-01 downloading (97MB so far) |
| **Disk Space** | ✓ OK | 94 GB free | ~10 GB needed total |

## Download Jobs

### Job A: NSIDC Sea-Ice Concentration (Priority 1)

**What:** Daily sea-ice concentration files (the model's target variable)  
**Source:** NOAA/NSIDC G02202 v6 CDR  
**Period:** 2008-01-01 to 2018-12-31 (11 years)  
**Files:** ~4,018 daily NetCDF files  
**Size:** ~1 GB total  
**Grid:** 332×316 polar stereographic (EPSG:3412)  

**Script:** `scripts/data/download_nsidc_sic.py`  
**Output:** `data/data/sic/{YYYY}/sic_pss25_{YYYYMMDD}_F17_v06r00.nc`  
**Authentication:** None required (public NOAA endpoint)

**Progress:**
- ✓ 2008: 366/366 files complete
- ⏳ 2009: 142/365 files (38% done)
- 2010-2018: Not started yet

**Expected Behavior:**
- Some files will fail validation (sensor gaps, known outages) → marked as missing
- Script automatically retries failed downloads (3 attempts)
- Missing dates are logged to `data/data/sic/MISSING_DATES.json`
- A handful of missing days (5-20) across the 11-year span is **normal and acceptable**

**Known Issues:**
- Occasional validation warnings for corrupted downloads → auto-retry handles these
- 2009-04-08 already marked as missing (sensor gap)

### Job B: ERA5 Atmospheric Forcing (Priority 2)

**What:** Atmospheric and sea-surface forcing variables  
**Source:** ECMWF Copernicus Climate Data Store (CDS)  
**Period:** 2008-01 to 2018-12 (132 months)  
**Files:** 132 monthly NetCDF files  
**Size:** ~5-8 GB total  
**Variables:** `u10`, `v10`, `t2m`, `sst` (4 variables)

**Script:** `scripts/data/download_era5_forcing.py`  
**Output:** `data/data/raw/era5/{YYYY}/era5_forcing_{YYYYMM}.nc`  
**Authentication:** `~/.cdsapirc` (already configured)

**Progress:**
- ⏳ 2008-01: Downloading (97 MB of ~153 MB downloaded)
- 2008-02 through 2018-12: Queued

**Expected Behavior:**
- Each monthly request queues at CDS → can take minutes to hours
- Requests processed sequentially (CDS throttles parallel requests)
- Script automatically aggregates 6-hourly data → daily means
- Fully resumable: interrupt with Ctrl-C and re-run to continue

**CDS Queue Times:**
- Typical: 1-5 minutes per request
- Peak hours: 5-20 minutes per request
- Total estimated time: 3-8 hours for all 132 months

## Monitoring Progress

### Quick Check

Run the status monitor script:

```bash
cd /Users/nawdddep/Documents/iceberg_models/iceberg_models-main/seaice_forecast
./CHECK_DOWNLOAD_STATUS.sh
```

This shows:
- File counts per year for both sources
- Total progress (X/Y files)
- Disk space available
- Process running status

### Detailed Logs

**NSIDC process:**
```bash
ps aux | grep download_nsidc_sic.py
```

**ERA5 process:**
```bash
ps aux | grep download_era5_forcing.py
```

### Check Downloaded Files

**NSIDC SIC:**
```bash
find data/data/sic -name "*.nc" | wc -l
ls -lh data/data/sic/2009/  # Current year
```

**ERA5:**
```bash
find data/data/raw/era5 -name "*.nc" | wc -l
ls -lh data/data/raw/era5/2008/  # Current year
du -sh data/data/raw/era5/
```

## What Happens Next

### When Downloads Complete

Both scripts will:
1. Write a summary report to stdout/logs
2. Generate a report file:
   - NSIDC: `data/data/sic/MISSING_DATES.json`
   - ERA5: `data/data/raw/era5/FAILED_MONTHS.json` (if any failures)
3. Exit with code 0 (success) or 1 (partial failure)

### Acceptance Verification

Once both downloads complete, run these checks:

#### Check 1: SIC File Count
```bash
find data/data/sic -name "*.nc" | wc -l
# Expected: ~4,000-4,015 (allow for small number of sensor gaps)
```

#### Check 2: SIC Grid Dimensions
```bash
venv/bin/python -c "
import xarray as xr, glob
f=sorted(glob.glob('data/data/sic/2015/*.nc'))[0]
ds=xr.open_dataset(f)
print(f)
print('Dimensions:', dict(ds.sizes))
print('Variables:', list(ds.data_vars))
"
# Expected: Dimensions should show 332×316 grid
```

#### Check 3: ERA5 Variables
```bash
venv/bin/python -c "
import xarray as xr, glob
f=sorted(glob.glob('data/data/raw/era5/2015/*.nc'))[0]
ds=xr.open_dataset(f)
print(f)
print('Variables:', list(ds.data_vars))
"
# MUST show: ['u10', 'v10', 't2m', 'sst']
# MUST NOT show: 'msl' (mean sea level pressure - wrong variable)
```

#### Check 4: ERA5 Coverage
```bash
find data/data/raw/era5 -name "era5_forcing_*.nc" | wc -l
# Expected: 132 files (12 months × 11 years)
```

#### Check 5: Missing Dates Report
```bash
cat data/data/sic/MISSING_DATES.json | jq 'length'
# Expected: 5-20 missing days (normal)
# If > 100: something went wrong, investigate
```

## If Something Goes Wrong

### Downloads Interrupted

Both scripts are **fully resumable**. Simply re-run:

```bash
# Resume NSIDC download
cd /Users/nawdddep/Documents/iceberg_models/iceberg_models-main/seaice_forecast
venv/bin/python scripts/data/download_nsidc_sic.py --start-year 2008 --end-year 2018

# Resume ERA5 download  
venv/bin/python scripts/data/download_era5_forcing.py --start-year 2008 --end-year 2018
```

Scripts will skip already-downloaded files automatically.

### ERA5 CDS Authentication Fails

If ERA5 download fails with authentication error:
1. Check `~/.cdsapirc` exists and contains valid credentials
2. Verify credentials at: https://cds.climate.copernicus.eu/
3. Update `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY
   ```

### NSIDC Download Fails

If NSIDC download fails completely (not just individual dates):
1. Test endpoint manually:
   ```bash
   curl -I https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2015/sic_pss25_20150115_F17_v06r00.nc
   ```
   Should return HTTP 200.
2. Check network connectivity
3. Re-run the script (it will resume from where it stopped)

### Disk Space Full

If disk space runs out mid-download:
1. Free up space (need ~10 GB total for both downloads)
2. Re-run scripts (they resume automatically)
3. Partial files are cleaned up automatically

## Timeline Estimate

**Optimistic (good network, minimal CDS queue):** 4-6 hours  
**Realistic (typical conditions):** 6-10 hours  
**Pessimistic (slow network, busy CDS):** 10-18 hours

### Breakdown:
- NSIDC SIC: 2-4 hours for ~4,000 files  
  (Currently: 3-5 seconds per file average)
- ERA5: 3-8 hours for 132 months  
  (1-5 minutes queue + 2-3 minutes download per month)

## What Was NOT Downloaded

Per task constraints, these were **deliberately excluded**:

1. **Ocean currents (GLORYS):** Already on disk (46 GB, 2008-2018, verified intact)
2. **Years 2019-2026:** Partial longitude coverage only (-180° to -150°), unusable for this model
3. **Synthetic/mock data:** Strictly disabled (downloaders raise errors instead of falling back)

## Data Sources and Verification

### NSIDC Sea-Ice Concentration
- **Product:** NOAA/NSIDC Climate Data Record (CDR) v6, Product G02202
- **Sensor:** SSMIS on DMSP F17 satellite (covers 2006-2023)
- **Algorithm:** NASA Team algorithm on polar stereographic grid
- **Resolution:** 25km
- **Grid:** EPSG:3412 (Antarctic Polar Stereographic), 332×316 cells
- **URL pattern:** `https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/{YYYY}/sic_pss25_{YYYYMMDD}_F17_v06r00.nc`
- **Validation:** Each file is opened with xarray to verify NetCDF structure and SIC variable presence

### ERA5 Reanalysis
- **Product:** ECMWF ERA5 Reanalysis (single-levels)
- **Dataset ID:** `reanalysis-era5-single-levels`
- **Variables:**
  - `10m_u_component_of_wind` → `u10`
  - `10m_v_component_of_wind` → `v10`
  - `2m_temperature` → `t2m`
  - `sea_surface_temperature` → `sst`
- **Region:** Southern Ocean south of 50°S ([-50, -180, -90, 180])
- **Temporal:** 6-hourly timesteps aggregated to daily means
- **Resolution:** ~31km (0.25° at ERA5 native resolution)
- **Validation:** Each file is opened with xarray to verify presence of all 4 variables

## Background Processes

Both downloads are running as background processes managed by Kiro:

**NSIDC process ID:** `term_1789400071070_mph08ypeuv`  
**ERA5 process ID:** `term_1789400085767_pg50ikp1as`

These will continue running even if you close this session. To check their status:

```bash
ps aux | grep download_nsidc_sic.py
ps aux | grep download_era5_forcing.py
```

To manually stop them (not recommended unless there's a problem):

```bash
pkill -f download_nsidc_sic.py
pkill -f download_era5_forcing.py
```

## Next Steps (After Downloads Complete)

1. **Verify all acceptance criteria** (see "Acceptance Verification" above)
2. **Review missing dates:** Check `MISSING_DATES.json` to ensure <100 missing days
3. **Run regridding pipeline:** `scripts/data/prepare_phase2.py` to create processed datasets
4. **Train model:** `scripts/train_with_real_data.py`

## Questions or Issues?

- Script locations: `seaice_forecast/scripts/data/download_*.py`
- Downloader implementations: `seaice_forecast/src/seaice_forecast/data_processing/downloaders/`
- Status monitor: `seaice_forecast/CHECK_DOWNLOAD_STATUS.sh`
- This document: `seaice_forecast/DOWNLOAD_STATUS_AND_INSTRUCTIONS.md`

---

**Last updated:** 2026-09-14 21:15  
**Status:** ✓ Both downloads running smoothly
