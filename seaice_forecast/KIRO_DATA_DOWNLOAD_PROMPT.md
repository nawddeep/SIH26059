# Task: Download the missing training data for the Antarctic sea-ice forecasting model

## Context

Repo root: `/Users/nawdddep/Documents/iceberg_models/iceberg_models-main`
Work inside: `seaice_forecast/`
Python env: **use `seaice_forecast/venv/bin/python`** — it already has `xarray`, `h5py`, `cdsapi`, `copernicusmarine`, `earthaccess`. The system `python3` does **not** have `h5py` and cannot open these NetCDF files.

The model trains on **7 variables**, defined in
`src/seaice_forecast/data_processing/dataset_real.py:65`:

```python
["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]
```

I audited what is on disk. Current state:

| Variable | Source | Status |
|---|---|---|
| `sic` | NSIDC G02202 v6 | **MISSING — nothing on disk** |
| `wind_u`, `wind_v` | ERA5 `u10`,`v10` | present |
| `air_temp` | ERA5 `t2m` | **MISSING — `msl` was downloaded instead** |
| `sst` | ERA5 `sst` | **MISSING — `msl` was downloaded instead** |
| `current_u`, `current_v` | GLORYS `uo`,`vo` | present, 2008–2018 circumpolar |

**Target years: 2008–2018 inclusive (11 years).** Do not download other years — see Constraint 1.

## Your job — exactly two downloads

### Job A — NSIDC sea-ice concentration (the target variable)

This is the top priority. Without it there is no label and nothing can train.

Use the **existing, working** downloader — do not write a new one:
`src/seaice_forecast/data_processing/downloaders/nsidc.py` (class `NSIDCDownloader`)

I verified the endpoint is live and needs **no authentication**:

```
https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2015/sic_pss25_20150115_F17_v06r00.nc
→ HTTP 200
```

- Hemisphere: `south`
- Every day from `2008-01-01` to `2018-12-31` (~4,018 files)
- Output dir: `data/data/sic/{YYYY}/`  ← note the doubled `data/data`, that is the real layout in this repo
- Sensor F17 covers this whole span; the class already falls back across F17/F18 and v06r00/v06r01 filename patterns.
- Expected total: well under 1 GB.

Write a runner script at `scripts/data/download_nsidc_sic.py` that:
- accepts `--start-year` / `--end-year` (default 2008 / 2018) and `--output-dir`
- is **resumable** — skip any date whose file already exists and is >50 KB (`download_date` already does this)
- logs progress with `tqdm` per year
- collects failed dates into a list and, at the end, writes `data/data/sic/MISSING_DATES.json` and prints a summary count
- does **not** abort the whole run on a single missing day (there are known gaps in the passive-microwave record, e.g. the 1987–88 outage and scattered sensor-transition days; a handful of missing days in 2008–2018 is normal and acceptable)

### Job B — ERA5 atmospheric forcing, with the correct variables

`src/seaice_forecast/data_processing/downloaders/era5.py` is **already correct** — do not change its variable list. It requests exactly what we need:

```python
ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",              # the missing air_temp
    "sea_surface_temperature",     # the missing sst
]
ANTARCTIC_AREA = [-50, -180, -90, 180]
```

The files currently on disk contain `u10, v10, msl` — they came from some other ad-hoc path, not this downloader. So: **just run this downloader.** `~/.cdsapirc` already exists, so CDS auth is configured.

- Call `ERA5Downloader.download_month(year, month)` for every month of 2008–2018 (132 requests).
- It already aggregates 6-hourly → daily means and skips existing files >100 KB, so it is resumable.
- Write the runner at `scripts/data/download_era5_forcing.py` with the same `--start-year`/`--end-year`/resumable/summary behaviour as Job A.
- Output to `data/data/raw/era5/{YYYY}/`.
- Expected total: roughly 5–8 GB.
- **CDS queues requests.** A month can sit in queue for minutes to hours. Run months sequentially, log the queue position the client reports, and make the script safe to Ctrl-C and re-run. Do not parallelise beyond 2 concurrent CDS requests — the API throttles and will start rejecting.

## Do NOT do these things

1. **Do not download ocean currents.** GLORYS `uo`/`vo` for 2008–2018 is already on disk at `data/data/{YYYY}/currents_GLORYS12_*.nc` — 4.2 GB per year, 46 GB total, circumpolar, verified intact (365 daily steps, no gaps, sane ranges). Re-downloading would waste ~46 GB and many hours. Leave it alone.
2. **Do not touch years 2019–2026.** Those files exist but cover only a 30° longitude sector (`lon −180..−150`) instead of circumpolar, so they are unusable for this model. They are a separate problem; do not try to "fix" or re-download them in this task.
3. **Do not invent, synthesise, interpolate, or mock any data.** `nsidc.py` deliberately raises `FileNotFoundError` rather than falling back to synthetic data — preserve that behaviour. If a download fails, record it as missing and move on.
4. **Do not delete anything** under `data/`. There is existing raw data and iceberg position data there.
5. **Do not run the regridding/training pipeline.** This task is download only. Producing `data/processed/regridded/daily/` is a separate follow-up.

## Constraints

- Disk: 95 GB free. Your two downloads need <10 GB combined. Check with `df -h` before starting and bail out with a clear message if under 15 GB free.
- Network: both jobs are long-running. Make both scripts resumable and idempotent so an interrupted run costs nothing.
- Credentials already configured: `~/.cdsapirc` (CDS/ERA5) and `~/.copernicusmarine/` (not needed for this task). NSIDC needs none.

## Acceptance criteria — verify before you report done

Run these and show me the output:

1. SIC file count is ~4,018 (allow a small number of known gaps):
   ```bash
   find seaice_forecast/data/data/sic -name "*.nc" | wc -l
   ```
2. A SIC file opens and has the expected CDR grid — **shape must be 332×316**:
   ```bash
   seaice_forecast/venv/bin/python -c "
   import xarray as xr, glob
   f=sorted(glob.glob('seaice_forecast/data/data/sic/2015/*.nc'))[0]
   ds=xr.open_dataset(f); print(f); print(dict(ds.sizes)); print(list(ds.data_vars))"
   ```
3. ERA5 now actually contains `t2m` and `sst` — this is the check that the original download failed:
   ```bash
   seaice_forecast/venv/bin/python -c "
   import xarray as xr, glob
   f=sorted(glob.glob('seaice_forecast/data/data/raw/era5/2015/*.nc'))[0]
   ds=xr.open_dataset(f); print(f); print(list(ds.data_vars))"
   ```
   The variable list **must** include `t2m` and `sst`. If it shows `msl`, the download used the wrong path — stop and tell me.
4. ERA5 covers all 132 months of 2008–2018, and each year's data is daily (not 6-hourly) after aggregation.
5. Print the contents of `MISSING_DATES.json` and state plainly how many days are missing and why.

## Report back

Tell me: files downloaded per source, total bytes, any missing dates with the reason, and whether all 4 acceptance checks passed. If something blocked you, say what and stop — do not work around it by substituting different data.
