# Task: Train an iceberg drift-prediction model (Antarctic)

## The one logistical thing to get right first

The forcing data (winds + ocean currents) is **66 GB of NetCDF** and lives on the
main machine. Do NOT try to copy it to a laptop.

Instead the pipeline is split in two:

| Stage | Where | Output |
|---|---|---|
| A. Feature extraction | main machine (has the 66 GB) | one CSV, ~5-20 MB |
| B. Model training | any laptop | trained model |

Stage A samples wind and current **only at the iceberg positions**, which turns
66 GB into a few megabytes. Stage B then trains on a laptop in minutes with no
NetCDF and no GPU.

If you are on the laptop and `drift_training_data.csv` already exists, skip to
Stage B.

---

## Stage A — build the feature table (run on the main machine)

Repo root: `/Users/nawdddep/Documents/iceberg_models/iceberg_models-main`
Python: **use `seaice_forecast/venv/bin/python`** (has xarray + h5py; system
python3 does not and cannot open these files).

### A1. Parse the iceberg tracks

Source: `seaice_forecast/data/data/raw/iceberg_positions/updated7_consol/*.csv`
(647 files, one per iceberg; also `iceberg_positions/` with 645 more — use both,
de-duplicate on (berg_id, date)).

Format — note the date encoding, it is NOT ISO:

```
date,nic_1,nic_2,nic_3,size_1,size_2
2011170,-66.48,142.96,1,10,2
```

- `date`  = `YYYYDDD`, year + day-of-year. `2011170` -> 2011-06-19.
            Parse with `datetime.strptime(str(d), "%Y%j")`.
- `nic_1` = latitude  (negative, Southern Hemisphere)
- `nic_2` = longitude
- `nic_3`, `size_1`, `size_2` = flags/size class — keep `size_1`, `size_2` as
            features, ignore `nic_3`.
- `berg_id` = the filename stem (e.g. `b09e`).

### A2. Build displacement targets

For each berg, sort by date. For each consecutive pair exactly **1 day apart**
(drop any gap != 1 day — tracks have holes):

```
u_berg = (lon[t+1] - lon[t]) * 111.32 * cos(radians(lat[t])) * 1000 / 86400   # m/s east
v_berg = (lat[t+1] - lat[t]) * 110.57 * 1000 / 86400                          # m/s north
```

Handle the ±180 dateline: if `|dlon| > 180`, wrap by ±360 before converting.

**Sanity filter — this matters.** Icebergs drift at roughly 0-1 m/s. Drop any
row where `sqrt(u_berg^2 + v_berg^2) > 2.0` m/s; those are tracking errors or
mis-identified bergs, and they will dominate an MSE loss if left in.

### A3. Sample the forcing at each position

For each surviving row, read the forcing at that date and position:

- Wind: `seaice_forecast/data/data/raw/era5/{YYYY}/era5_forcing_{YYYYMM}.nc`
        variables `u10`, `v10` (also `t2m`, `sst` available)
- Currents: `seaice_forecast/data/data/{YYYY}/currents_GLORYS12_{YYYY}-01-01_{YYYY}-12-31_chunk0.nc`
        variables `uo`, `vo` (take `depth=0`)

Use `ds.sel(latitude=lat, longitude=lon, method="nearest")` after selecting the
time step. **Open each yearly/monthly file once and process all rows for that
period** — reopening per row will take hours instead of minutes.

Coverage caveat: only **2008-2018** is circumpolar. Years 2019+ cover only a 30°
sector (lon -180..-150) and must be excluded unless the berg is inside it.

### A4. Write the table

`drift_training_data.csv`, one row per berg-day:

```
berg_id, date, lat, lon, size_1, size_2,
u_wind, v_wind, u_curr, v_curr,
u_berg, v_berg          <- targets
```

Report: rows written, date range, number of distinct bergs, rows dropped by the
speed filter. Then copy this single CSV to the laptop.

---

## Stage B — train the model (runs on any laptop)

Input: `drift_training_data.csv` only. No NetCDF, no GPU, no venv from the main
machine — just pandas + scikit-learn (+ xgboost or lightgbm if available).

### B1. Split chronologically, never randomly

Iceberg tracks are time series and consecutive rows are highly correlated. A
random split leaks future into past and produces a meaningless score.

```
train: earliest 70% of dates
val:   next 15%
test:  latest 15%
```

### B2. Baseline you must beat — the "2 % rule"

The standard empirical model for iceberg drift is:

```
u_pred = u_curr + 0.02 * u_wind
v_pred = v_curr + 0.02 * v_wind
```

Icebergs travel with the current plus roughly 2 % of the wind speed. Compute its
error on the test split **first**. This is the number that decides whether the ML
model is worth anything — treat it exactly like persistence in a forecasting
task.

Also report a trivial baseline: `u_pred = u_curr, v_pred = v_curr` (pure current
advection, no wind).

### B3. Models to train

Two targets (`u_berg`, `v_berg`), so train two regressors, or one multi-output.

1. **Linear regression** on `[u_wind, v_wind, u_curr, v_curr]`.
   This is the interesting one scientifically: the fitted coefficients should
   recover approximately 1.0 on the current terms and ~0.02 on the wind terms.
   **Report the coefficients** — if they match the physics, that is a strong
   result to show a reviewer.
2. **Gradient boosting** (XGBoost / LightGBM / sklearn's
   `HistGradientBoostingRegressor`) on the same features plus `lat`, `lon`,
   `size_1`, `size_2`, and month-of-year. This can capture regional and seasonal
   effects the linear model cannot.
3. Optional: add Coriolis-relevant features — `sin(lat)`, wind speed magnitude,
   current speed magnitude.

Keep it simple. Do NOT start with the PINN in
`iceberg_drift_model/src/iceberg_drift/models/pinn.py` — get a working GBM
result first, then consider it.

### B4. Metrics to report

Per model, on the **test** split:

- RMSE and MAE on `u_berg` and `v_berg` separately (m/s)
- **24-hour position error in km** — the number a reviewer actually cares about:
  `sqrt((u_err*86400)^2 + (v_err*86400)^2) / 1000`
- Skill score vs the 2 % rule: `1 - RMSE_model / RMSE_2pct` (positive = better)

Produce one table: rows = {current-only, 2 % rule, linear, GBM}, columns =
{RMSE_u, RMSE_v, 24h position error km, skill vs 2 %}.

### B5. Save

- Model to `models/drift_gbm.pkl` (joblib)
- Metrics to `output/drift_metrics.json`
- A scatter plot of predicted vs actual 24 h displacement

---

## Constraints

- **Never random-split** time series data (see B1).
- **Never train on the test split**, and compute any scaling/normalisation
  statistics from the **train split only**.
- Do not invent or interpolate iceberg positions. If forcing data is missing for
  a row, drop the row and count it.
- Report honestly: if the GBM does not beat the 2 % rule, say so. A physics
  baseline beating ML is a legitimate and publishable finding, and claiming
  otherwise will not survive questioning.

## Acceptance

Print, and show me:

1. Row count, berg count, and date range of `drift_training_data.csv`
2. The linear model's fitted coefficients (do they recover ~1.0 current, ~0.02 wind?)
3. The full comparison table from B4
4. One sentence: does the ML model beat the 2 % rule, and at what 24 h position error?
