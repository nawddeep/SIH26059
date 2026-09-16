# Task: Wire the fuel model into A* routing

## Why this matters

The problem statement (SIH 26059) asks for routes that are "safe and
**fuel-efficient**." Right now the router only optimises distance + ice risk —
there is no fuel term anywhere in the cost function. This task adds one.

## What already exists — use it, do not reimplement it

**Fuel physics** — `seaice_forecast/src/seaice_forecast/fuel/consumption.py`

```python
from seaice_forecast.fuel import fuel_per_cell_array, speed_in_ice_array, FuelConfig

fuel_per_cell_array(distance_km, sic_array, polar_class="PC4") -> np.ndarray
```

Takes a SIC grid, returns fuel in **tonnes per cell** for a given Polar Class.
Already verified: monotonic in SIC, 4.3x fuel penalty in heavy ice at SIC=0.95
vs open water, POLARIS-class-aware (PC1 vessels burn less than UNCLASSED at the
same SIC because they hold higher speed). Do not touch this file - it is
correct and tested.

**Risk grid** — `seaice_forecast/src/seaice_forecast/risk/polaris.py`, function
`risk_ice_array(sic, polar_class, config)`. Already wired into routing (see
below). Also do not touch.

**The router to extend** —
`seaice_forecast/src/seaice_forecast/routing/pathfinding.py`, function
`build_cost_grid()` (lines 55-102). Currently:

```python
def build_cost_grid(
    sic_grid, polar_class="PC4", distance_weight=1.0, risk_weight=1.0,
    land_mask=None, pixel_size_km=25.0, config=None,
) -> np.ndarray:
    ...
    risk_grid = risk_ice_array(sic_clean, polar_class=polar_class, config=config)
    base_dist = 1.0
    cost_grid = (distance_weight * base_dist) + (risk_weight * 10.0 * risk_grid)
    if land_mask is not None:
        cost_grid[land_mask] = np.inf
    cost_grid[nan_mask] = np.inf
    return cost_grid
```

## The change

Add fuel as a third weighted term, so the cost function becomes:

```
C(r,c) = distance_weight * base_cost
       + risk_weight     * 10.0 * risk_cost(r,c)
       + fuel_weight      * fuel_cost(r,c)
```

### 1. Extend `build_cost_grid()`

Add parameters `fuel_weight: float = 1.0` and `fuel_config: Optional[FuelConfig] = None`.
Inside the function:

```python
from seaice_forecast.fuel import fuel_per_cell_array

fuel_grid = fuel_per_cell_array(
    distance_km=pixel_size_km, sic=sic_clean,
    polar_class=polar_class, config=fuel_config,
)
```

**Normalise the fuel grid before combining it.** Fuel is in tonnes (roughly
0.9-4.5 per 25 km cell from the existing table); risk and distance are
dimensionless O(1) terms. Adding raw tonnes will make `fuel_weight` dominate or
vanish depending on vessel size, which defeats the point of having three
independently-tunable weights. Normalise so the fuel term is also O(1) at a
sensible reference:

```python
FUEL_REFERENCE_T_PER_CELL = 0.875   # open-water fuel/cell at v_ref (see FuelConfig defaults)
fuel_cost = fuel_grid / FUEL_REFERENCE_T_PER_CELL   # ~1.0 in open water, ~4.3 in heavy ice
cost_grid = (distance_weight * base_dist
             + risk_weight * 10.0 * risk_grid
             + fuel_weight * fuel_cost)
```

(If `fuel_config` overrides `base_rate_t_per_km` or `v_ref_kn`, recompute the
reference from that config rather than hardcoding 0.875 - call
`fuel_per_cell(pixel_size_km, sic=0.0, polar_class, config=fuel_config)` once to
get it.)

Keep land-masking and NaN-masking exactly as they are, applied last, after all
three terms are combined.

### 2. Update the docstring

State the new formula and that `fuel_weight=0` recovers the exact current
behaviour (needed for backward compatibility - see Acceptance below).

### 3. Thread the new parameters through the CLI / demo entrypoint

Find where `build_cost_grid` is currently called (`demo_phase_f.py` and/or
`scripts/evaluation/run_ablation.py` - grep for `build_cost_grid(`) and add
`--fuel-weight` and `--polar-class` (if not already present) as CLI args,
mirroring how `--risk-weight` and `--distance-weight` are already exposed. Also
add these three weights to `configs/phase_f.yaml` under a `routing:` block
alongside the existing `distance_weight` / `risk_weight`.

### 4. Produce the comparison that matters for the demo

Using a real SIC field from `data/processed/regridded/daily/2017/*.npz` (index 0
is the SIC channel) and a start/goal pair that crosses meaningful ice (reuse the
coordinates already in `configs/phase_f.yaml`), compute three routes:

| route | weights |
|---|---|
| distance-only | `distance_weight=1, risk_weight=0, fuel_weight=0` |
| risk-optimal (current default) | `distance_weight=1, risk_weight=2, fuel_weight=0` |
| fuel-optimal (new) | `distance_weight=1, risk_weight=0, fuel_weight=2` |

For each route report: total distance (km, via `path_distance()`), total fuel
(tonnes - sum `fuel_grid` along the path cells), and total transit time (hours -
use `transit_time_hours()` from the fuel module cell-by-cell along the path).

**The result to look for:** the fuel-optimal route should be longer in distance
than the distance-only route, but burn less total fuel, because it avoids heavy
ice where speed collapses and transit time triples. If fuel-optimal comes out
identical to distance-only, something is wrong (the fuel grid is not varying
enough, or the weight is too small to matter) - do not report success in that
case, investigate instead.

Save the three routes as a comparison plot (reuse the plotting style already in
`demo_phase_f.py`) to `output/phase_f/fuel_route_comparison.png`, and the
numbers to `output/phase_f/fuel_route_comparison.json`.

## Constraints

- Do not modify `fuel/consumption.py` or `risk/polaris.py` - they are already
  correct and tested.
- `fuel_weight=0` must reproduce today's routes exactly (bitwise-identical cost
  grid) - this is how you avoid silently changing behaviour for anyone already
  using `build_cost_grid()`.
- Land and NaN masking must still set cost to `np.inf`, applied after summing
  all three weighted terms, not before.
- Do not add a new dependency for this - `fuel_per_cell_array` already exists
  and is vectorised.

## Acceptance

Show me:

1. `build_cost_grid(sic, fuel_weight=0, ...)` produces the same array as before
   your change (run it against a saved reference grid, or diff against git
   stash of the pre-change output).
2. The three-route comparison table (distance, fuel, transit time per route).
3. Confirmation that the fuel-optimal route burns less fuel than distance-only,
   even though it is not the shortest path - with the actual numbers, not just
   "yes".
4. Any existing tests in `tests/` that touch `pathfinding.py` still pass.
