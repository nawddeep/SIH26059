# How a route is chosen

A reader should be able to answer *"why did the algorithm choose this route?"*
from this repository alone. This document is the cost function as implemented in
[`shipNavigation/backend/app/route_engine.py`](../shipNavigation/backend/app/route_engine.py),
not an idealised version of it.

## The search

A* over a lat/lon grid, 8-connected, with a great-circle heuristic
(`geo.haversine` to the destination). The heuristic never overestimates true
remaining distance, so A* returns the optimal path for the cost function below.

Land cells are excluded outright rather than penalised. Canal waterways (Suez,
Panama, Kiel) are inserted as explicit edges with their own traversal cost.

After the search the path is simplified with **Douglas–Peucker** and smoothed
with **Chaikin** corner-cutting. Both operate on geometry only and never move a
point onto land.

## The cost of one edge

Each edge starts as great-circle distance in metres and is then **multiplied**
by a series of factors. Multiplication rather than addition is deliberate: it
keeps every term dimensionless and scale-free, so a term cannot dominate merely
because distance happens to be large.

```
edge = great_circle_distance
     × objective_factor      (depends on optimizeFor)
     × polaris_factor        (always applied)
     × ice_resistance_factor (skipped in fuel mode - see below)
```

### 1. Objective factor — depends on `optimizeFor`

| mode | factor | rationale |
|------|--------|-----------|
| `distance` | `1.0` | pure geometry |
| `time` | `target_speed / SOG` | speed over ground includes current projected along track, plus 3% of along-track wind |
| `fuel` | `max(0.25, (v_water/v_target)³ × time_factor) × ice_fuel_multiplier` | engine load scales roughly cubically with water-relative speed |
| `safety` | `1 + 0.25·(conc/10) + weather + iceberg` | see below |

**Safety mode** adds:
- `weather = max(0, (wind_knots − 12) × 0.04)` — gale penalty above 12 kn
- an iceberg term: for every berg within 60 nm, `+3.5 × (60 − distance_nm)/60`

The berg positions come from the **trained drift model** (`current_iceberg_positions()`).
If that model is unavailable the list is empty and the term contributes nothing —
the route is never bent around invented bergs.

### 2. POLARIS factor — always applied

```
edge ×= 1 + 2.5 × polaris_risk
```

`polaris_risk` is the IMO POLARIS Risk Index Outcome for **this hull class** at
this cell's ice concentration (`seaice_forecast/risk/polaris.py`).

This is the term that makes the same ice field produce different routes for
different ships. Concentration alone is hull-agnostic: 70% ice is routine for a
PC2 icebreaker and prohibitive for an unclassed vessel. Risk ranges 0–1, so the
factor ranges 1× to 3.5×.

### 3. Ice resistance factor

Applied **except in fuel mode**, where the physical fuel model already prices ice
through speed collapse and power rise. Applying both would double-count ice and
let the heuristic rather than the physics pick the route.

| vessel | concentration | factor |
|--------|---------------|--------|
| non-polar | > 45% | `1 + 0.35 × conc` (up to ~36×) |
| non-polar | 15–45% | `1 + 0.08 × (conc − 15)` |
| polar / icebreaker | any | `1 + 0.015 × conc` |

Heavy ice is penalised steeply for non-polar hulls but never made infinite, so
routes to polar stations remain possible rather than failing outright.

## Normalisation

Every factor is a **multiplier ≥ 0.25**, so no term can drive an edge negative or
to zero. The weights above were chosen so that in heavy pack the ice terms reach
roughly the same order of magnitude as a large detour — a route will accept a
few hundred extra nautical miles to avoid impassable ice, but not several
thousand.

The fuel multiplier is capped in practice by the physical model: burn per km
rises about 4.3× between open water and full ice cover for a PC4 hull, and the
heuristic ice factor would reach ~32× over the same range, which is why the two
are never applied together.

## What the route reports back

Every route carries an `explanation` block naming the dominant cost term:

```json
{
  "objective": "lowest exposure to ice, weather and icebergs",
  "distanceKm": 1722.2,
  "estimatedFuelTonnes": 73.3,
  "estimatedTimeHours": 80.99,
  "maxIceRisk": 0.541,
  "fractionOfTrackInIce": 0.138,
  "icebergsTracked": 12,
  "closestIcebergNm": 510.6,
  "icebergIntersections": 0,
  "iceDataSource": "UNet+ConvLSTM forecast (NSIDC CDR v6)",
  "dominantCostTerm": "open-water distance dominated: no term materially diverted the track",
  "reason": "Optimised for lowest exposure to ice, weather and icebergs. ...",
  "caveat": "Costed against historical reanalysis, not a live feed. Advisory only ..."
}
```

## Refusals

The router refuses rather than guessing:

- **No sea-ice field** → `get_sea_ice_concentration` raises. There is no
  simulated fallback; a route costed against a fabricated ice field is
  indistinguishable from a real one, and that is the failure most likely to put
  a vessel somewhere it should not be.
- **Waypoints at the same position** → `RouteError` naming the pair.
- **Fewer than two waypoints** → `RouteError`.
- **Destination unreachable** (interior land, no navigable path) → `RouteError`.

## Known defect: objectives do not reliably win on their own metric

`POST /api/route-alternatives` plans the same passage under all four objectives
and **checks whether each one actually wins on the metric it optimises**. On a
representative Antarctic Peninsula passage it does not:

```
the 'safety' objective scores 0.541 on peak POLARIS risk, worse than 'distance' at 0.373
the 'safety' objective scores 128.2 on distance in ice,  worse than 'fuel'     at 67.8
the 'fuel'   objective scores  37.6 on fuel burn,        worse than 'distance' at 37.0
the 'time'   objective scores 38.28 on duration,         worse than 'fuel'     at 38.14
```

Two distinct problems.

**Serious — the safety objective.** It returns a track 119% longer that carries
*higher* peak POLARIS risk and nearly double the distance in ice. A route
labelled "Safest" that loses on every safety metric is worse than no safety
option at all, because the label is what a reader trusts. The cost weights were
rebalanced to make POLARIS dominant and cap the weather term, which changed the
ordering but did not fix this case. Root cause is not yet established.

**Minor — the other three.** `_leg_path` returns the great-circle track directly
when `optimize_for == "distance"` and the track is land-free, bypassing A*
entirely. The grid-based objectives can therefore find paths a few tenths of a
percent better on distance and duration than "distance" mode itself. A grid
artefact, not a modelling error.

Until the safety objective is fixed, **treat its label as unreliable** and read
the comparison table rather than the label. The API says so itself: a non-empty
`warnings` array means an objective lost on its own metric.

This is surfaced rather than hidden because the comparison endpoint exists to
make disagreements between objectives visible, and the most important
disagreement it found was the system contradicting itself.

## Limits worth stating

- The cost weights are **engineering judgement calibrated against published
  vessel performance**, not learned from routing outcomes. There is no dataset of
  optimal Antarctic routes to fit them to.
- The ice field is **historical reanalysis** ending 2018-12-31. Every route is a
  hindcast.
- The iceberg term uses **predicted positions at a single time**, not swept
  volumes over the passage duration.
- This is **advisory**. It does not replace an ice navigator, official ice charts
  or the master's judgement.
