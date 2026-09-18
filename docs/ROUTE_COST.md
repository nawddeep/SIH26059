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
*higher* peak POLARIS risk and nearly double the distance in ice.

Partial root cause, established by sampling both corridors:

```
corridor    lat     lon   conc%  polaris  penalty
direct    -66.0   -66.0    72.7    0.584     6.32x   severe patch
western   -66.0   -79.0     0.2    0.000     1.01x   ice-free
```

A* is not malfunctioning. The direct corridor crosses a 6.32x penalty patch, and
the safety objective routes around it through ice-free water. **The objective
minimises risk-weighted total exposure** - it accepts 128 nm in light ice rather
than 68 nm that includes a heavy patch. As a trade-off that is defensible.

What this does **not** explain is why peak POLARIS risk ends up *higher* (0.541
vs 0.373). Minimising a sum can raise a maximum, and for ice navigation "safest"
arguably means minimising the worst moment a hull must survive rather than the
integral of exposure - a hull either can take the ice in front of it or it
cannot, and an average does not help. But that reasoning is a hypothesis, not a
demonstrated cause.

Ruled out by experiment: iceberg proximity (nearest berg 509 nm from the track),
the weather term (disabling it made the route *worse* - 1503 nm at 0.854 peak
risk, so weather was constraining rather than causing), and an ice-field mismatch
between A* and the reported metrics (both call `get_sea_ice_concentration`).

The likely fix is a minimax formulation for the safety objective - bound the
worst cell the route may cross, rather than integrating exposure - but that is a
different search, not a weight change, and it has not been implemented.

**Also real — the fuel objective.** On the same passage `fuel` burns 37.6 t
while `time` burns 37.0 t: 1.6% worse on the metric it exists to minimise. Both
are grid-searched, so this is not the great-circle artefact below. The ice fuel
multiplier and the cubic water-speed term are applied to the same edge, and the
combination appears to over-penalise light-ice cells that `time` crosses
happily. Not yet isolated.

**Artefact, not a defect — "distance" winning on distance.** `_leg_path` returns
the great-circle track directly when `optimize_for == "distance"` and the track
is land-free, bypassing A*. It is therefore not grid-constrained while every
other objective is, and beats them by tenths of a percent on distance and
duration. Comparisons against `distance` are not like-for-like.

### What was ruled out, and how

Both defects were investigated by experiment rather than argument:

| Hypothesis | Test | Result |
|---|---|---|
| Iceberg term diverts the track | measured berg distance | nearest is 509 nm — not it |
| Weather term dominates | disabled it | route got **worse** (1503 nm, 0.854) — it was constraining, not causing |
| A* and metrics use different ice | traced both | same `get_sea_ice_concentration` — not it |
| Post-hoc smoothing moves the track | compared raw vs smoothed | 0.843 → 0.854, only 0.011 — not it |
| Metrics depend on sampling density | densified 4× | unchanged — not it |
| A better grid path does not exist | ran all objectives through A* | **it does**: distance/fuel/time reach 0.373 in ~25 cells, safety takes 198 to reach 0.843 |

That last row is the important one. A 0.373 path is reachable on the grid and
three objectives find it, so the safety search is not constrained by geometry -
its objective is wrong.

Raising the risk weight makes it **worse**, which is the signature: 40× gives
0.541, 99999× gives 0.854. A cost that sums risk over a track will always trade
one severe cell for many mild ones. That is the wrong trade for ice, where a
hull either survives the worst cell it meets or it does not.

The fix is a **bottleneck (minimax) search** - order the frontier by the worst
risk on the path, then by distance. An implementation was attempted and reverted:
it did not find the known-better 0.373 path and the reason was not isolated, and
shipping an unproven search is worse than shipping a documented defect.

### Encoded as tests

`shipNavigation/backend/tests/test_routing_safety.py::TestObjectivesWinTheirOwnMetric`
asserts that each objective wins on its own metric. Both failures are marked
`xfail` with these reasons, so they stay visible in every test run rather than
living in a review comment.

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
