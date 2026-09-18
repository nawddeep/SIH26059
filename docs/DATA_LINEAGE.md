# Data lineage

Every dataset from raw source to the route planner, with the preprocessing that
sits between. Written so an auditor can trace any number back to a file.

```
raw source  →  download  →  quality checks  →  regridding  →  normalisation
                                                                    ↓
route planner  ←  evaluation  ←  model  ←  training dataset  ←──────┘
```

---

## Sea-ice concentration

| | |
|---|---|
| **Source** | NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, **version 6** (NSIDC-0051 / G02202) |
| **Variable** | `cdr_seaice_conc` — daily sea-ice concentration, fraction 0–1 |
| **Spatial resolution** | 25 km |
| **Grid** | Southern Hemisphere polar stereographic, **EPSG:3412**, 332 × 316 |
| **Temporal resolution** | daily |
| **Date range** | 2008-01-01 → 2018-12-31 |
| **Access** | NASA Earthdata (`scripts/download_sic_data.py`, prompts for credentials) |

**Preprocessing:** values outside 0–1 are flags (pole hole, land, missing) and
are masked rather than clipped. Land is excluded by a 0/1 mask
(`data/processed/land_ocean_mask_ps25.npy`), leaving **83,019 ocean cells** of
the 104,912 in the grid.

---

## Atmospheric forcing

| | |
|---|---|
| **Source** | ECMWF **ERA5** reanalysis, single levels |
| **Variables** | `u10`, `v10` (10 m wind components), `t2m` (2 m air temperature) |
| **Native resolution** | 0.25° × 0.25°, hourly |
| **Used as** | daily means, regridded to the 25 km EPSG:3412 grid |
| **Date range** | 2008-01-01 → 2018-12-31 |
| **Access** | Copernicus CDS API (`cdsapi`), credentials in `~/.cdsapirc` |

---

## Ocean forcing

| | |
|---|---|
| **Source** | Copernicus Marine **GLORYS12V1** global ocean reanalysis |
| **Variables** | `thetao` (sea surface temperature), `uo`, `vo` (surface currents) |
| **Native resolution** | 1/12° (~8 km), daily |
| **Used as** | surface level, regridded to the 25 km EPSG:3412 grid |
| **Date range** | 2008-01-01 → 2018-12-31 |
| **Access** | `copernicusmarine` toolbox, CMEMS credentials via environment variables |

---

## Iceberg fixes

| | |
|---|---|
| **Source** | **US National Ice Center** Antarctic iceberg tracking |
| **Variables** | berg id, date, latitude, longitude, size (two axes, nautical miles) |
| **Spatial resolution** | point observations |
| **Temporal resolution** | irregular, roughly weekly per berg |
| **Date range** | 2008-01-13 → 2018-12-21 |
| **Volume** | 14,913 fixes across **103 distinct bergs** |
| **Coordinate system** | WGS84 lat/lon |

**Derived features** (`iceberg-drift/data/drift_trainable_rich.csv`, 27 features):
observed drift `u_berg`/`v_berg` from consecutive fixes; co-located ERA5 wind and
GLORYS currents; lagged velocities (1, 2, 3 steps); rolling means; `frac_moving`;
track age; and seasonal `sin`/`cos` of month.

---

## Regridding

All forcing is regridded from its native grid onto the sea-ice grid
(EPSG:3412, 25 km, 332 × 316) so every channel shares one geometry.

- **Method:** nearest-neighbour via a KD-tree built on 3-D Cartesian coordinates
  rather than lat/lon, so the ±180° seam and the pole converge correctly. A naive
  lat/lon distance puts points either side of the antimeridian far apart when
  they are neighbours.
- **Output:** one `.npz` per day, `data[7, 332, 316]`, channels in fixed order:

  `["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]`

---

## Missing-data handling

| Case | Treatment |
|---|---|
| Missing day in the archive | the sample is dropped; the dataset never interpolates across a gap |
| Land cells | masked out of both loss and metrics; never counted as correct zeros |
| Pole hole | part of the land/ocean mask, excluded |
| Non-finite forcing | the day is rejected at quality-check time (`iceberg-drift/.../quality.py`) |
| Missing berg feature | row dropped via `dropna(subset=features)` — never imputed, because an imputed wind speed drives a physical prediction |

A scan of all 2,829 training files found **0 non-finite values**; maximum absolute
z-score after normalisation was 18.1.

---

## Normalisation

Computed on the **training split only**, then applied unchanged to validation and
test. Stored in `data/processed/normalization_stats.json`.

- **SIC is not normalised.** It is already 0–1 and is the prediction target.
- **The other six channels** are z-scored per variable: `(x − mean) / std`.

This file has been silently overwritten before with a single scalar pair, which
made every lookup miss, normalisation quietly no-op, and the network see raw
Kelvin (~273 against a 0–1 target) — producing a fully ice-covered ocean that
looked plausible and was entirely wrong. `model_bridge._load_norm_stats()` now
validates the shape and falls back to a reference copy rather than proceeding.

---

## Splits

| split | range | purpose |
|---|---|---|
| **train** | 2008-01-01 → 2015-12-31 | model fitting, climatology, normalisation stats |
| **validation** | 2016-01-01 → 2016-12-31 | checkpoint selection, LR schedule, early stopping |
| **test** | 2017-01-01 → 2018-12-31 | held-out evaluation only |

Chronological, not random. Random splits leak across a time series: adjacent days
are nearly identical, so a randomly held-out day sits between two training days
and is trivially predictable.

Climatology baselines are built from the **train split only**, for the same reason.

---

## Model inputs

| model | input | output |
|---|---|---|
| Sea-ice forecaster | `[7 days, 7 channels, 332, 316]` normalised | `[332, 316]` concentration 0–1 |
| Iceberg drift | 27 tabular features per berg-fix | moving/stationary, then `u`/`v` in m/s |

---

## Downstream

```
sea-ice forecast ──► POLARIS risk (IMO MSC.1/Circ.1519)
                 └─► ice-aware fuel model (Riska speed degradation)
                                    │
iceberg drift ──────────────────────┼──► A* cost grid ──► route + explanation
                                    │
telemetry gateway ──────────────────┴──► display only, not a model input
```

Telemetry is deliberately **not** an input. It is current, while this archive
ends 2018-12-31, so anything derived from both would pair a real position with an
eight-year-old environment.

---

## Operational ingestion

`scripts/ingest.py` chains the existing downloaders, quality checks and
regridding into one entry point:

```bash
python scripts/ingest.py --check                              # readiness, no network
python scripts/ingest.py --plan  --start .. --end ..          # what is missing
python scripts/ingest.py --fetch --start .. --end ..          # acquire
```

Current readiness on this machine: **2 of 3 sources**. ERA5 and CMEMS
credentials are present; NSIDC needs a NASA Earthdata login in `~/.netrc`.

`--fetch` deliberately does not run an unattended loop. Each source has its own
rate limits and queueing behaviour — the CDS API queues requests for minutes to
hours, and a naive retry loop gets an account throttled. The per-source scripts
handle that properly and the orchestrator points at them.

**The blocker is not plumbing, it is validation.** These models were trained on
NSIDC CDR v6, a climate record tuned for consistency across decades. The
near-real-time product (NSIDC-0081) is a different instrument calibration, ERA5T
is subject to revision, and GLORYS12 is reanalysis rather than the
analysis-forecast product. Feeding any of those to these weights would produce
numbers that look right and are not comparable to anything in the published
evaluation. Operational use needs revalidation against the near-real-time
products first.

---

## Provenance caveats

- Every figure in this system is **historical reanalysis**. Nothing is real-time.
- Reanalysis is itself a model product, not direct observation. ERA5 and GLORYS12
  assimilate observations but interpolate where none exist — sparse in the
  Southern Ocean.
- NSIDC CDR v6 is a *climate* record, tuned for consistency over time rather than
  day-to-day accuracy. A near-real-time product (NSIDC-0081) has different
  characteristics, so a model trained here would need revalidation against it.
