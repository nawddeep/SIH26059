# Ship Route Planner

An interactive sea-route planner: pick multiple waypoints (ports, Antarctic research
stations, or open-ocean coordinates), set speed and departure time, and get a
great-circle geodesic route drawn on a map with distance, duration, and ETA.

Built for Smart India Hackathon 2026. Works offline-friendly (all spatial data is
bundled locally; only base-map tiles are fetched from the internet).

Staged plan from the brief: **Stage 3 (current)** — land avoidance, ECA distance
and chokepoint crossings are computed server-side and the client renders the real
results. Routes avoid land where the straight great circle would cut across it,
and the results panel reports genuine ECA distance and named crossings.

## Features

- **Multi-waypoint routing** — order-aware; find a port/station by name (NGA World
  Port Index + Antarctic stations), type `lat, lon` for open-ocean legs, or click the
  map to add a point.
- **Land-avoiding geodesic routing** — every leg is tried as a pure great circle
  (Wincenty on WGS84); if it would cross land, an A\* search over the 0.5°
  navigable-water grid produces a detour that is Douglas–Peucker simplified and
  shortcut onto natural water lines, so the result reads as a curve, not a stair
  case of grid cells.
- **Route statistics** — total distance (nm) measured along the drawn polyline,
  duration at chosen speed, ETA (UTC), **distance inside Emission Control Areas**
  and **named chokepoint crossings** — all real values from the backend.
- **Live map** — MapLibre GL with OpenStreetMap / CARTO-light basemaps (toggle),
  numbered waypoint pins, directional route arrows, plus ECA polygons (orange) and
  chokepoint lines (red dashes) as reference overlays.
- **Working India → Antarctica** — established research stations (Bharati, Maitri,
  and 9 others) are included as routable waypoints, route stays highway-clear of
  land because open-ocean legs are used as-is when unobstructed.
- **Antimeridian-safe** — great circles that cross the 180° line are split into
  continuous per-world-copy parts, so the line never wraps across the whole map.
- Recomputes and redraws immediately (350 ms debounce) on every waypoint
  add/remove/reorder/edit and on speed/departure changes.

## Quick start

Backend (FastAPI + Python 3.10+):

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn app.main:app --port 8600 --host 127.0.0.1
```

Frontend (Node 18+):

```bash
cd frontend
npm install
npm run dev
```

Or just run the repo script (creates `.venv`, installs deps, starts both):

```bash
./start.sh
```

Then open the URL printed by Vite (default `http://localhost:5173/`; it falls back
to the next free port if 5173 is busy — Vite binds `localhost` on this machine).
The dev server proxies `/api/*` to the backend on port 8600, so no CORS setup is
needed. Routes are computed by the backend (`POST /api/route`); the browser only
sends waypoints/speed/departure and renders the returned polyline + metadata.

## API

Routing is **server-side**: the client posts waypoints + settings, the backend
computes land-avoiding legs with ECA distance and chokepoint crossings, and the
client renders whatever geometry comes back.

| Method | Path                     | Description                                    |
| ------ | ------------------------ | ---------------------------------------------- |
| GET    | `/api/health`            | Readiness check                                |
| GET    | `/api/ports?q=mumbai`    | Search ports / stations by name or alias       |
| GET    | `/api/geocode?lat=&lon=` | Reverse-geocode into a waypoint                |
| GET    | `/api/ecas`              | ECA polygons (GeoJSON) for the map overlay     |
| GET    | `/api/chokepoints`       | Chokepoint lines / cape points (GeoJSON)       |
| POST   | `/api/route`             | Land-avoiding route engine                     |

`POST /api/route` body:

```json
{
  "waypoints": [
    { "id": "1", "label": "Chennai", "lat": 13.08, "lon": 80.27, "isPort": true },
    { "id": "2", "label": "Bharati", "lat": -69.41, "lon": 76.19, "isPort": false }
  ],
  "speedKnots": 15,
  "departureTimeUTC": "2026-09-12T08:00:00Z"
}
```

Response: `legs[]` (each with `from`, `to`, `path` as `[lat, lon]` pairs,
`distanceNm`, `durationHours`, and real `distanceInEcaNm` + `crossings`), plus
`totalDistanceNm`, `totalDistanceInEcaNm`, `totalDurationHours`, `etaUTC`, and the
deduped, route-ordered top-level `crossings`.

## Architecture

```
frontend/
  src/store.jsx       React context; debounced POST /api/route, holds route/loading/error
  src/api.js          Backend client (ports, geocode, route, ecas, chokepoints)
  src/geo.js          WGS84 geodesics (Wincenty) — antimeridian unwrap/split for rendering
  src/components/     MapView (MapLibre), RoutePlannerPanel, WaypointRow, ResultsPanel
backend/              FastAPI routing service
  app/geo.py          Wincenty / haversine geodesics (pure Python)
  app/mask.py         0.5° land–ocean mask + nearest-water snapping (built once at
                      startup, cached)
  app/datastore.py    ports/stations search, reverse geocode, ECA + chokepoint loaders
  app/route_engine.py per-leg router: geodesic → A* over the sea grid → simplify →
                      shortcut → safe-splice; ECA distance + chokepoint crossings
  app/main.py         REST API
  data/               ports.json (3,813 pts), ecas.json, chokepoints.json, ne_110m_land.geojson
```

### Route pipeline (current — server side)

1. **Great circle** between every pair of waypoints on WGS84 (Wincenty inverse +
   direct), sampled every 20 nm.
2. **Land check** on the precomputed 0.5° land–ocean grid: if the geodesic never
   enters a land cell, use it as-is (keeps open-ocean legs — e.g. the Southern
   Ocean to Antarctica — straight and cheap, identical to Stage 2).
3. Otherwise **A\*** over the navigable-water grid with geodesic edge weights and a
   geodesic distance-to-goal heuristic. Arctic penalty above 65°N forces trade
   lanes; hard navigation cap at 78°N blocks polar shortcuts.
4. **Simplify** (Douglas–Peucker), then a greedy **safe-shortcut** pass splices the
   grid path onto natural water lines (with re-segmentation wherever a shortcut
   would reground), and the result is resampled to 20 nm.
5. **Crossings**: test the final leg polyline against the named chokepoint table —
   cape proximity for capes, exact line intersection (plus a one-cell tolerance,
   `STRAIT_TOL_DEG = 0.4`, for coarse-grid alignment) for straits and canals.
   Dedupe and order them across the whole route.
6. **ECA distance**: intersect each final leg polyline with the IMO ECA polygons
   and sum the geodesic length of the parts that fall inside any of them.
7. **Stage E**: duration and ETA are derived on top; isolated so Phase-2 (currents,
   ice) can swap timing without touching the geometry.

## Data sources

- **Ports** — NGA World Port Index, `WPI` updated edition (`UpdatedPub150.csv`,
  public domain), 3,802 ports + Unlocode aliases.
- **Antarctic stations** — 11 research stations (Bharati 76.19E 69.41S, Maitri,
  McMurdo, Davis, Casey, Mawson, etc.) merged into the port DB.
- **Land** — Natural Earth `ne_110m_land` (public domain).
- **Basemap tiles** — OpenStreetMap tile servers and CARTO basemaps (require
  internet at runtime).

## Known limitations

The routing grid is intentionally coarse (0.5°) so the graph stays fast and small
and builds at startup in under a second. Consequences:

- **Suez and Panama Canals are not transit-able** — they are narrower than a grid
  cell, so Mediterranean↔Indian-Ocean traffic routes around the Cape of Good Hope
  (which is correctly reported as a crossing). Same for the Straits of Magellan:
  Punta Arenas is effectively trapped in an "enclosed sea" at this resolution.
- **Straits are detected within one grid cell** of their reference line
  (`STRAIT_TOL_DEG`) rather than by exact throat crossing — real transits always
  register, but borderline near-transits can too.
- **ECA distance is indicative** — ECA polygons are simplified boxes and the
  distance inside each is exact geodesic length clipped to the polygon, so values
  are reasonable, not regulatory-grade.
- Grid-search distance overshoots by roughly 20% on detoured legs versus a naive
  straight line (that's the honest cost of going around land).
- Antarctic and other straight open-ocean legs are used as-is (never detoured), so
  India → Antarctica looks exactly like a Stage-2 geodesic with 0 nm of ECA.

Other notes:

- **Atomic verification** for the current stage: Dhamra → Chennai ≈ **599.7 nm**;
  Mumbai → Colombo ≈ **950 nm** (avoiding India's landmass through the Maldovic
  corridor rather than cutting the subcontinent); Singapore → Colombo reports
  **Malacca Strait**; Hamburg → Antwerp reports **664 nm in ECA**; Mumbai →
  Rotterdam rounds the Cape and reports **Cape of Good Hope**; Mumbai → San
  Francisco splits at the 180° meridian (~75.7°N) and reports **316 nm in ECA**;
  Parangipettai → Bharati stays a straight geodesic with **0 nm ECA**, confirming
  the skip-pathfinding fast path.
- DRY: leg crossing names come from `data/chokepoints.json`; ECA polygons from
  `data/ecas.json` (Home Office / IMO box approximations).
- Route optimization is distance-only; time/weather/current-aware routing is the
  next phase (the backend already isolates timing as "Stage E", and the UI hints at
  it).

## Project structure

```
backend/app/            Python package (engine + API)
backend/data/           Bundled spatial data assets
frontend/src/           React app (components + store + api client)
instruciton.md          Competition brief this app implements
```