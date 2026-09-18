# Data and model fallback hierarchy

What happens when a model or dataset is unavailable. The governing rule:

> **A component that cannot answer says so. It never fabricates an answer that
> looks like a real one.**

A route costed against invented ice is indistinguishable from a route costed
against real ice. That is the failure most likely to put a vessel somewhere it
should not be, so degradation here is always visible.

## The hierarchy

```
         LIVE MODEL
              │ unavailable
              ▼
   LATEST VALID FORECAST                  (newest day in the processed archive)
              │ unavailable
              ▼
    ┌─────────────────────┐
    │  REFUSE             │  ← routing and risk stop here, deliberately
    │  (explicit error)   │
    └─────────────────────┘
              │
              ▼
      CLIMATOLOGY              ← available ONLY as a labelled comparison layer,
                                  never as a routing input
```

## Why climatology is not a routing fallback

The obvious hierarchy ends `… → CLIMATOLOGY → NO ROUTE`. This system deliberately
ends `… → REFUSE`.

Climatology is a **day-of-year long-term average**. Using it to cost a route for
a specific week says "ice is usually like this in mid-December" and presents it
with exactly the same confidence as a forecast. It is smooth, plausible and
silently wrong about the thing that matters — where the ice edge is *this* week.

That is the same class of failure as the simulated ice fields removed from this
codebase, differing only in that the fabrication is statistically motivated.
For a safety-related decision the distinction does not help the master.

Climatology remains valuable as an **evaluation baseline** (it is one of the two
baselines every model is scored against) and could be exposed as a clearly
labelled comparison layer. It is not wired into route costing.

## Per-component behaviour

| Component | Live | Degraded | Never |
|---|---|---|---|
| **Sea-ice field** (`/api/sea-ice`) | forecaster output | empty FeatureCollection, `source: "unavailable"` + error | simulated field |
| **Route costing** (`get_sea_ice_concentration`) | `sic_at_point` from the model | **raises** | substituted field |
| **POLARIS risk** | computed from the live field | unavailable with the field it depends on | default or assumed risk |
| **Iceberg positions** (router) | trained drift model | empty tuple — the safety term contributes nothing | placeholder coordinates |
| **Iceberg layer** (`/api/icebergs`) | drift model trajectories | empty list | mock bergs |
| **Ocean currents / wind** | GLORYS12 / ERA5 | empty, `source: "unavailable"` | synthetic vector field |
| **Weather heatmap** | — | always empty, `"no real data source for this layer"` | the synthesised hazard index it used to return |
| **Telemetry** | last-known values | `connected: false` | last known value presented as current |
| **Route** | full route + explanation | `RouteError` → HTTP 422 with a reason | a route with an invented cost basis |

## How a caller detects degradation

A rendering map is not evidence the models loaded — every data endpoint returns
a valid empty response rather than failing, precisely so the map survives. Check
explicitly:

```bash
curl localhost:8600/api/system-status
```

```json
{
  "sea_ice": "live", "iceberg": "live", "polaris": "live",
  "fuel": "live", "routing": "live", "telemetry": "no_data",
  "environment_data_date": "2018-12-31",
  "realtime": false, "allLive": true, "degraded": []
}
```

Every component is **probed** during the request — `live` means it was exercised,
not that a file exists on disk. `degraded` lists anything that failed.

Each route also carries `explanation.iceDataSource`, naming the field it was
costed against. If that ever reads `unavailable`, the route is not trustworthy
and the string says so.

## Refusal cases in routing

| Condition | Result |
|---|---|
| Sea-ice model unavailable | raises — no route |
| Fewer than two waypoints | `RouteError` |
| Two waypoints at the same position | `RouteError` naming the pair |
| Destination unreachable (interior land, no path) | `RouteError` |
| Search exceeds 800,000 expansions | `ValueError` — refuses rather than returning a partial path |

## Tested

`shipNavigation/backend/tests/test_routing_safety.py`:

- `test_missing_ice_model_refuses_rather_than_inventing`
- `test_ice_data_source_never_claims_synthetic`
- `test_no_bergs_yields_no_iceberg_term_not_fake_ones`
- `test_land_locked_destination_is_refused_not_faked`
- `test_identical_start_and_destination`
