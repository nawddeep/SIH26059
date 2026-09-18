# Antarctic Navigation Decision Support System

**Smart India Hackathon 2026 — Problem Statement 26059**
Ministry of Earth Sciences (MoES) · National Centre for Polar and Ocean Research (NCPOR)

AI-enabled forecasting of Antarctic sea-ice concentration, iceberg trajectory
prediction, and safe / fuel-efficient route planning for research vessels —
plus the shipboard data-acquisition pipeline that gets vessel telemetry ashore
over a satellite link that barely exists below 70°S.

---

## The four models

| Model | Type | What it does | Measured performance |
|-------|------|--------------|----------------------|
| **Sea-ice forecaster** | Trained (PyTorch) | U-Net + ConvLSTM, **next-day** concentration over a 332×316 EPSG:3412 grid | At +1d beats climatology ~2.5× (0.044 vs 0.110) but is **beaten by persistence** (0.020). Not usable beyond +1d — see below |
| **Iceberg drift** | Trained (scikit-learn) | Two-stage HistGradientBoosting — moving/stationary gate, then u/v regressors | **3.91 km RMS** 24 h position error, 96.7% within 10 km, 9.5% skill vs constant-velocity |
| **POLARIS ice risk** | Deterministic | IMO MSC.1/Circ.1519 Risk Index Outcome | Validated by property tests — a published standard has no accuracy figure |
| **Ice-aware fuel model** | Deterministic | Speed collapse × power ramp, per Polar Class | Validated by property tests |

Two hold weights fitted from data. Two implement published formulas and have
**no learned parameters** — labelled as such throughout rather than dressed up
with accuracy numbers they cannot have.

Exported as loadable artifacts in [`model_exports/`](model_exports/) with
SHA-256 hashes and round-trip samples in `manifest.json`.

---

## Data acquisition — the shipboard scraper

**[`telemetry-gateway/`](telemetry-gateway/)** — this is the data-scraping half
of the system, and it solves a problem most polar prototypes ignore.

Below roughly 70°S there is little or no geostationary satellite coverage, so
research vessels fall back on narrowband LEO links that are slow, costly per
megabyte, and unavailable for stretches at a time. A research vessel already
has GPS, AIS, a weather station and engine telemetry feeding its bridge
displays. What it lacks is a way to get that data *off the ship*.

```
ship instruments ──► gateway/ingest.py ──► gateway/buffer.py ──► gateway/sender.py
   NMEA 0183 +        concurrent feed       durable SQLite        priority drain,
   JSON feeds         clients, parsed       store-and-forward     ACK-gated
                      and prioritised              │
                                                   ▼  satellite uplink
                                          shore/shore_listener.py
                                          receives · acknowledges · de-duplicates
```

| Component | File | Role |
|-----------|------|------|
| **Scraper / ingest** | `gateway/ingest.py` | Four concurrent TCP clients reading ship instruments; NMEA 0183 parsing with checksum validation, JSON parsing, priority tagging, independent per-feed reconnect with backoff |
| **Buffer** | `gateway/buffer.py` | Durable SQLite queue (WAL) — the "store" in store-and-forward |
| **Sender** | `gateway/sender.py` | Drains the queue highest-priority-first over the constrained link |
| **Shore receiver** | `shore/shore_listener.py` | Acknowledges by sequence number, de-duplicates, optionally forwards to the dashboard |

**Priority tiers**, assigned at ingestion: position fixes and alerts are HIGH,
routine weather/engine telemetry MED, redundant fix-quality detail LOW. After an
outage a position fix queued a second ago still beats a low-priority row queued
an hour ago.

**Delivery is at-least-once, never at-most-once.** A record is marked sent only
after shore returns `ACK <seq>`. This matters: a successful `sendall()` proves
only that bytes reached a local kernel socket buffer — under a dead link the
write still succeeds. Measured before the acknowledgement was added, **6 of 578
records were silently lost while the gateway reported success.**

Verified on a Raspberry Pi 3: zero data loss across a total link outage
(backlog held, drained in priority order on restore) and across a `SIGKILL` of
the sender mid-flight, with shore de-duplicating the re-sends.

Link degradation is applied at the OS level with `tc`/`netem`, filtered to the
uplink port so the ship's own instrument bus is unaffected — as it would be
aboard a real vessel. See [`telemetry-gateway/README.md`](telemetry-gateway/README.md).

---

## Web application

**[`shipNavigation/`](shipNavigation/)** — FastAPI backend + React/MapLibre frontend.

- Route planning with A*, ice-aware cost grid, Douglas–Peucker and Chaikin smoothing
- Live map layers: sea ice, ice risk, ocean currents, wind, icebergs
- **Model dashboard** at `/model-tests.html` — loads each `.pkl` from disk, runs it,
  and shows the output, timings, SHA-256 and held-out evaluation tables
- Vessel telemetry strip fed by the gateway (display only)

There are **no simulated data sources**. Every endpoint returns an empty field
tagged `unavailable` rather than substituting invented data, and routing raises
rather than costing a route against a fabricated ice field.

---

## Layout

```
seaice_forecast/     sea-ice forecasting pipeline, training, evaluation
iceberg-drift/       iceberg drift model, training, evaluation
model_exports/       the four exported artifacts + manifest.json
shipNavigation/      FastAPI backend + React frontend + model dashboard
telemetry-gateway/   shipboard data acquisition and satellite uplink
integration/         cross-component glue
```

---

## Verify every claim in this README

```bash
./verify.sh
```

Runs both test suites (96 tests), loads each exported artifact from disk and
calls `predict()` on it, checks the SHA-256 against `manifest.json`, and prints
the held-out evaluation numbers **read straight from the files the training runs
wrote** — so the figures above can be checked against their source rather than
taken on trust. Exits non-zero if anything fails.

The four `.pkl` artifacts are committed (28 MB), so this works on a fresh clone.

---

## Running it

```bash
# web app + model dashboard
cd shipNavigation && ./start.sh
#   http://localhost:5173                  route planner
#   http://localhost:5173/model-tests.html model dashboard

# telemetry gateway
cd telemetry-gateway/shore   && ./run_all.sh
cd telemetry-gateway/gateway && ./run.sh
```

The models need Python 3.9 with torch, joblib, scikit-learn and pandas;
`start.sh` prefers `seaice_forecast/venv`, which has them. A backend started
without that environment reports the models as unavailable — check
`/api/model-status` rather than assuming a rendering map means they loaded.

---

## Honest limitations

These are stated plainly because a decision-support tool that overstates itself
is worse than one that does not exist.

- **The sea-ice forecaster is beaten by persistence, and is a next-day model
  only.** At +1d it beats climatology by roughly 2.5× (0.044 vs 0.110 MAE over
  the ice zone), so it has learned real structure, but persistence scores 0.020
  and is the stronger baseline.

  Beyond +1d it should not be used. The shipped checkpoint was trained with
  `forecast_horizon=1`; driving it autoregressively compounds its own error at
  every step, and by +5d it is beaten by climatology as well. Measured with
  `scripts/evaluation/rollout_comparison.py`:

  | lead | model | persistence | climatology |
  |------|-------|-------------|-------------|
  | +1d  | 0.044 | **0.020** | 0.110 |
  | +3d  | 0.103 | **0.036** | 0.110 |
  | +5d  | 0.147 | **0.046** | 0.108 |
  | +7d  | 0.182 | **0.055** | 0.108 |
  | +14d | 0.231 | **0.084** | 0.111 |

  The older `baseline_comparison.py` reports better multi-day figures, but it
  applies a single forward pass at every horizon — grading a one-day forecast
  against truth a week later. Those numbers are only valid at +1d. Both scripts
  and both result files are committed so the difference can be inspected.

  A direct multi-horizon architecture exists in
  `src/seaice_forecast/models/multi_horizon.py` and is the right way to serve
  leads beyond +1d; it is not what the shipped checkpoint trained. Training also
  hits a documented non-finite-loss problem — see
  [`seaice_forecast/OPEN_PROBLEM.md`](seaice_forecast/OPEN_PROBLEM.md).
- **Nothing here is real-time.** The processed archive ends **2018-12-31**, so
  every forecast is historical reanalysis.
- **Vessel telemetry is display-only.** It is deliberately not an input to the
  models: telemetry is current while the ice archive is not, so anything derived
  from both would pair a real position with an eight-year-old environment.
- **The 90 MB training checkpoints are not in this repository**, though the four
  28 MB exported artifacts in `model_exports/` are, so `./verify.sh` runs on a
  fresh clone. Retraining from scratch needs the checkpoints and the 82 GB raw
  archive, neither of which belongs in git.
