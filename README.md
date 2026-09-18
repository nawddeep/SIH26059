# Antarctic Navigation Decision Support System

**Smart India Hackathon 2026 — Problem Statement 26059**
Ministry of Earth Sciences (MoES) · National Centre for Polar and Ocean Research (NCPOR)

Forecasting of Antarctic sea-ice concentration, iceberg trajectory prediction,
and safe / fuel-efficient route planning for research vessels — plus the
shipboard data-acquisition pipeline that gets vessel telemetry ashore over a
satellite link that barely exists below 70°S.

**Two trained models and two published standards.** The iceberg drift predictor
and the sea-ice forecaster are fitted from data; POLARIS risk and the fuel model
implement a regulatory standard and a physical model respectively and have no
learned parameters. Calling all four "AI models" would be generous, so this
document does not.

That is a design choice, not a shortfall. POLARIS is the IMO's own Risk Index
Outcome table — the instrument that decides whether a given hull may enter given
ice. Replacing a regulatory standard with a neural network would make the system
worse, not more advanced.

---

## The four components

| Model | Type | What it does | Measured performance |
|-------|------|--------------|----------------------|
| **Sea-ice forecaster** | Trained (PyTorch) | U-Net + ConvLSTM, **next-day** concentration over a 332×316 EPSG:3412 grid | At +1d beats climatology 2.4× (0.047 vs 0.113) but is **beaten by persistence** (0.021), n=300. Next-day only — see below |
| **Iceberg drift** | Trained (scikit-learn) | Two-stage HistGradientBoosting — moving/stationary gate, then u/v regressors | **3.91 km RMS** 24 h position error, 96.7% within 10 km, 9.5% skill vs constant-velocity |
| **POLARIS ice risk** | Deterministic | IMO MSC.1/Circ.1519 Risk Index Outcome | Validated by property tests — a published standard has no accuracy figure |
| **Ice-aware fuel model** | Deterministic | Speed collapse × power ramp, per Polar Class | Validated by property tests |

Two hold weights fitted from data. Two implement published formulas and have
**no learned parameters** — labelled as such throughout rather than dressed up
with accuracy numbers they cannot have.

### On the sea-ice result

The headline is that the forecaster loses to persistence at +1d, and that is
stated first rather than buried. Two things are worth knowing alongside it.

**Persistence is the hard baseline at this horizon, not a trivial one.** Sea-ice
concentration changes little in 24 hours, so "tomorrow resembles today" is close
to optimal by construction. Beating it at short lead times is an open problem in
sea-ice forecasting, not a sign of a botched implementation. The model does beat
climatology by 2.4×, so it has learned real structure — it has simply not learned
enough to overtake a baseline that is very strong here.

**The model is weakest where it matters most.** Uncertainty measured on held-out
data shows near-perfect performance over open water (MAE 0.003) and its worst
performance in the marginal ice zone (MAE 0.097) — exactly where a vessel
operates. A single overall MAE of 0.019 hides that entirely, because 97% of the
domain is trivially predictable open water. That is the honest limitation, and
it is a more useful thing to know than the headline number.

**It is a next-day model and should not be used beyond that.** Driven
autoregressively it compounds its own error and is beaten by climatology from
+5d. A direct multi-horizon architecture exists at
`seaice_forecast/src/seaice_forecast/models/multi_horizon.py` and is the right
way to serve longer leads; it is not what the shipped checkpoint trained.

Exported as loadable artifacts in [`model_exports/`](model_exports/) with
SHA-256 hashes and round-trip samples in `manifest.json`.

---

## Shipboard Telemetry Acquisition & Store-and-Forward Gateway

**[`telemetry-gateway/`](telemetry-gateway/)** — the data-acquisition half of the
system, and the part most polar prototypes skip.

The pipeline is: **acquisition → parsing → prioritisation → persistent buffering
→ reliable transmission → acknowledgement → de-duplication.** Not scraping: the
ship's instruments are the source, and nothing here reads a screen or a web page.

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
| **Acquisition / ingest** | `gateway/ingest.py` | Four concurrent TCP clients reading ship instruments; NMEA 0183 parsing with checksum validation, JSON parsing, priority tagging, independent per-feed reconnect with backoff |
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

### Measured uncertainty

Confidence comes from held-out error, not assumption. `GET /api/uncertainty`:

| condition | expected error (MAE) | p90 |
|---|---|---|
| open water (SIC < 0.15) | **0.0031** | 0.0022 |
| marginal ice zone (0.15–0.50) | **0.0971** | 0.2247 |
| pack (0.50–0.85) | 0.0923 | 0.1692 |
| consolidated (0.85+) | 0.0797 | 0.1195 |

The steepest-gradient band carries **11.2×** the error of the flattest, so the
ice edge is an order of magnitude harder than the interior.

This is worth stating rather than averaging away: **the forecaster is
near-perfect over open water and least confident in the marginal ice zone —
exactly where a vessel operates.** Iceberg positions carry a 3.9 km envelope
(the model's own held-out RMS), so a predicted position is the centre of a
circle, not a point.

Neither is a calibrated probabilistic forecast. Both say how wrong the model has
been in conditions like these.

---

## Documentation

| Document | Covers |
|---|---|
| [`docs/DATA_LINEAGE.md`](docs/DATA_LINEAGE.md) | Every dataset from raw source to route planner: resolution, date range, preprocessing, splits, missing-data handling, coordinate systems |
| [`docs/ROUTE_COST.md`](docs/ROUTE_COST.md) | The A* cost function as implemented — every term, its weight, and why routes come out as they do |
| [`docs/FALLBACK.md`](docs/FALLBACK.md) | What happens when a model or dataset is unavailable, and why climatology is deliberately not a routing fallback |
| [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | Telemetry gateway: threats, mitigations, tests, and the gaps that remain |
| [`docs/models/sea_ice_model_card.md`](docs/models/sea_ice_model_card.md) | Sea-ice forecaster: architecture, training, evaluation, limitations, intended and unintended use |
| [`docs/models/iceberg_model_card.md`](docs/models/iceberg_model_card.md) | Iceberg drift predictor: same |

---

## Layout

```
seaice_forecast/     sea-ice forecasting pipeline, training, evaluation
iceberg-drift/       iceberg drift model, training, evaluation
model_exports/       the four exported artifacts + manifest.json
shipNavigation/      FastAPI backend + React frontend + model dashboard
telemetry-gateway/   shipboard data acquisition and satellite uplink
local-dashboard/     bridge console (React 19 + MapLibre), served by the real backend
integration/         cross-component glue
```

---

## Verify every claim in this README

```bash
./verify.sh                 # fast: documents vs stored results
./verify.sh --recompute     # slower: also re-derives the evaluation from the checkpoint
```

Runs all four test suites (122 tests, including routing-safety tests for
impassable ice, degraded data and degenerate requests, and an end-to-end integration test
that drives telemetry → shore → backend → models → risk → route → fuel), loads
each exported artifact from disk and
calls `predict()` on it, checks the SHA-256 against `manifest.json`, and then
checks that **every performance figure in this README and in `OPEN_PROBLEM.md`
matches the JSON the evaluation wrote** — 26 figures, and a mismatch fails the
run. Performance numbers are never hand-maintained in two places.

`--recompute` goes further: it re-derives the evaluation from the trained
checkpoint and compares the fresh numbers against the committed ones. That is
the only check that proves the stored results are what the model actually
produces rather than a stale file.

Also exposed at runtime: **`GET /api/system-status`** returns the live state of
every component and the date of the environmental data behind it.

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

  | lead | model | persistence | climatology | n |
  |------|-------|-------------|-------------|---|
  | +1d  | 0.045 | **0.020** | 0.107 | 120 |
  | +3d  | 0.106 | **0.036** | 0.107 | 116 |
  | +5d  | 0.151 | **0.047** | 0.107 | 111 |
  | +7d  | 0.177 | **0.055** | 0.107 | 104 |

  The older `baseline_comparison.py` reports better multi-day figures, but it
  applies a single forward pass at every horizon — grading a one-day forecast
  against truth a week later. Those numbers are only valid at +1d, where both
  scripts agree to rounding (0.0466 vs 0.0467 at n=300). Both scripts and both
  result files are committed so the difference can be inspected.

  Both evaluations were previously run on ~60 of roughly 700 available test
  samples, which produced numbers unstable enough to support a false conclusion:
  an earlier `OPEN_PROBLEM.md` claimed the model beat persistence at +5d and +7d.
  At n=300 it does not, and the correction is recorded there rather than quietly
  removed.

  A direct multi-horizon architecture exists in
  `src/seaice_forecast/models/multi_horizon.py` and is the right way to serve
  leads beyond +1d; it is not what the shipped checkpoint trained. Training also
  hits a documented non-finite-loss problem — see
  [`seaice_forecast/OPEN_PROBLEM.md`](seaice_forecast/OPEN_PROBLEM.md).
- **Nothing here is real-time.** The processed archive ends **2018-12-31**, so
  every forecast is historical reanalysis — real measured data from NSIDC, ERA5
  and GLORYS12, but not current. *Historical* and *simulated* are different
  claims: there is no invented data anywhere in this system, and `scripts/ingest.py`
  documents what acquiring current data would require and why the models would
  need revalidation first.
- **Vessel telemetry is display-only.** It is deliberately not an input to the
  models: telemetry is current while the ice archive is not, so anything derived
  from both would pair a real position with an eight-year-old environment.
- **The 90 MB training checkpoints are not in this repository**, though the four
  28 MB exported artifacts in `model_exports/` are, so `./verify.sh` runs on a
  fresh clone. Retraining from scratch needs the checkpoints and the 82 GB raw
  archive, neither of which belongs in git.
