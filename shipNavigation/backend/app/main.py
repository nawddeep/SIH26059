"""FastAPI backend for the Ship Route Planner.

POST /api/route        compute a sea route
GET  /api/ports?q=     ports autocomplete
GET  /api/geocode      reverse lookup for dropped pins / typed coords
GET  /api/ecas         ECA polygons as GeoJSON (for the basemap overlay)
GET  /api/chokepoints  chokepoint crossing lines as GeoJSON
GET  /api/health
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .datastore import Datastore
from .mask import LandMask
from .route_engine import RouteEngine, RouteError, build_waterway_midline


class Waypoint(BaseModel):
    id: str = ""
    label: str
    countryCode: Optional[str] = ""
    lat: float
    lon: float
    isPort: bool = True

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "countryCode": self.countryCode or "",
            "lat": self.lat,
            "lon": self.lon,
            "isPort": self.isPort,
        }


class RouteRequest(BaseModel):
    waypoints: list[Waypoint] = Field(min_length=2)
    speedKnots: float = Field(gt=0, description="1-24 kn in the UI; not hard-clamped server-side")
    departureTimeUTC: str = Field(default="", description="ISO 8601")
    optimizeFor: str = "distance"  # 'distance' | 'time' (@see phase 2)
    vesselType: str = Field(default="cargo", description="cargo, tanker, passenger, service, fishing, icebreaker, research, other")
    draftMeters: float = Field(default=10.0, description="vessel operational draft in meters")
    iceClass: str = Field(default="none", description="none, pc1_pc7, icebreaker")


def _parse_departure(raw: str) -> datetime:
    raw = (raw or "").strip()
    if not raw:
        return datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, f"invalid departureTimeUTC: {exc}") from exc
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


import uuid
from .weather_engine import get_route_weather


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Datastore()
    mask = LandMask()
    for cp in store.chokepoints:
        if cp["kind"] == "strait" and cp["line"]:
            mask.carve_water([(lat, lon) for lon, lat in cp["line"]])
    for w in store.waterways:
        mask.carve_water(build_waterway_midline(w))
    engine = RouteEngine(mask, store.ecas, store.chokepoints, waterways=store.waterways)
    app.state.datastore = store
    app.state.engine = engine
    app.state.routes = {}
    app.state.booted = True
    print("[api] ready", flush=True)
    yield


logger = logging.getLogger(__name__)

app = FastAPI(title="Ship Route Planner", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"ok": True, "booted": getattr(app.state, "booted", False)}


@app.post("/api/route")
async def compute_route(req: RouteRequest):
    try:
        engine: RouteEngine = app.state.engine
    except AttributeError:
        raise HTTPException(503, "route engine not ready")
    dep = _parse_departure(req.departureTimeUTC)
    try:
        res = engine.compute_route(
            [w for w in req.waypoints],
            speed_knots=req.speedKnots,
            departure_time_utc=dep,
            optimize_for=req.optimizeFor or "distance",
            vessel_type=req.vesselType,
            draft_meters=req.draftMeters,
            ice_class=req.iceClass,
        )
        route_id = str(uuid.uuid4())
        res["routeId"] = route_id
        res["departureTimeUTC"] = dep.strftime("%Y-%m-%dT%H:%M:%SZ")
        if hasattr(app.state, "routes"):
            app.state.routes[route_id] = res
            # Keep cache bounded to 50 routes
            if len(app.state.routes) > 50:
                oldest_key = next(iter(app.state.routes))
                app.state.routes.pop(oldest_key, None)
        return res
    except RouteError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/route-weather")
async def route_weather(
    routeId: str = Query(default=""),
    departureTime: str = Query(default=""),
    speedKnots: float = Query(default=15.0),
):
    dep = _parse_departure(departureTime)
    routes_cache = getattr(app.state, "routes", {})
    route_obj = routes_cache.get(routeId)
    
    if not route_obj:
        if not routeId and len(routes_cache) > 0:
            # Fall back to most recent computed route
            route_obj = list(routes_cache.values())[-1]

    if not route_obj:
        return {"samples": []}

    return get_route_weather(route_obj, dep)




@app.get("/api/ports")
async def search_ports(q: str = Query(default="", max_length=64)):
    return app.state.datastore.search_ports(q)


@app.get("/api/geocode")
async def geocode(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    return app.state.datastore.reverse_geocode(lat, lon)


@app.get("/api/ecas")
async def ecas():
    return app.state.datastore.ecas_geojson()


@app.get("/api/chokepoints")
async def chokepoints():
    return app.state.datastore.chokepoints_geojson()


@app.get("/api/icebergs")
async def icebergs(
    startTime: str = Query(default=""),
    endTime: str = Query(default=""),
    stepHours: float = Query(default=1.0, gt=0, le=24),
):
    from datetime import timedelta
    dep_start = _parse_departure(startTime)
    if endTime:
        dep_end = _parse_departure(endTime)
    else:
        dep_end = dep_start + timedelta(days=7)
    try:
        from .model_bridge import icebergs_with_trajectories_real
        return icebergs_with_trajectories_real(dep_start, dep_end, step_hours=stepHours)
    except Exception as exc:  # noqa: BLE001 - report, never 500 the map
        logger.warning("drift model unavailable: %s", exc)
        return []


@app.get("/api/sea-ice")
async def sea_ice(date: str = Query(default="")):
    """Sea-ice concentration from the trained U-Net+ConvLSTM forecaster.

    Returns an empty field with an error rather than substituting simulated
    ice. A map that silently shows invented ice is worse than a map that shows
    none: the first misleads a navigator, the second tells them to look
    elsewhere.
    """
    try:
        from .model_bridge import sea_ice_geojson_real
        return sea_ice_geojson_real(date or None)
    except Exception as exc:  # noqa: BLE001 - report, never 500 the map
        logger.warning("sea-ice model unavailable: %s", exc)
        return {"type": "FeatureCollection", "features": [],
                "error": f"{type(exc).__name__}: {exc}", "source": "unavailable"}


@app.get("/api/ocean-currents")
async def ocean_currents(date: str = Query(default="")):
    """Real GLORYS12 surface currents. No simulated substitute."""
    try:
        from .model_bridge import ocean_currents_geojson_real
        return ocean_currents_geojson_real(date or None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("currents data unavailable: %s", exc)
        return {"type": "FeatureCollection", "features": [],
                "error": f"{type(exc).__name__}: {exc}", "source": "unavailable"}


@app.get("/api/wind")
async def wind(date: str = Query(default="")):
    """Real ERA5 10 m wind. No simulated substitute."""
    try:
        from .model_bridge import wind_geojson_real
        return wind_geojson_real(date or None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wind data unavailable: %s", exc)
        return {"type": "FeatureCollection", "features": [],
                "error": f"{type(exc).__name__}: {exc}", "source": "unavailable"}


@app.get("/api/ice-risk")
async def ice_risk(date: str = Query(default=""), polarClass: str = Query(default="PC4")):
    """IMO POLARIS navigational ice risk, derived from the SIC forecast."""
    try:
        from .model_bridge import ice_risk_geojson_real
        return ice_risk_geojson_real(date or None, polar_class=polarClass)
    except Exception as exc:  # noqa: BLE001 - degrade like the sibling layers
        logger.warning("ice-risk model unavailable: %s", exc)
        return {"type": "FeatureCollection", "features": [],
                "error": "model unavailable", "source": "unavailable"}


@app.get("/api/model-tests")
async def model_tests():
    """Verifiable evidence for the model test dashboard.

    Loads each artifact, runs it, and returns the result alongside the held-out
    metrics from the training runs, so the page shows measurements rather than
    claims.
    """
    try:
        from .model_tests import run_model_tests
        return run_model_tests()
    except Exception as exc:  # noqa: BLE001
        logger.exception("model tests failed")
        return {"error": f"{type(exc).__name__}: {exc}", "components": []}


@app.get("/api/model-status")
async def model_status():
    """Which trained models are actually live behind the map.

    Every data endpoint returns an empty field rather than failing when a
    model is missing, so a rendering map is not evidence the models loaded.
    This runs them.
    """
    try:
        from .model_bridge import model_status as status
        return status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model status unavailable: %s", exc)
        return {
            "allLive": False, "liveComponents": [], "missingArtifacts": [],
            "components": {}, "error": f"{type(exc).__name__}: {exc}",
        }


@app.get("/api/weather-heatmap")
async def weather_heatmap():
    """Removed: this layer had no real data source behind it.

    It synthesised a hazard index from invented temperature, wind and current
    fields. Kept as an endpoint so the frontend gets an empty layer rather than
    a 404, but it will stay empty until a real product backs it.
    """
    return {"type": "FeatureCollection", "features": [],
            "error": "no real data source for this layer", "source": "unavailable"}


# ---------------------------------------------------------------- explorer
# Interactive counterparts to /api/model-tests. Each one loads a .pkl from
# model_exports/ and runs it against inputs the reader chose, so the dashboard
# shows the artifact answering a question rather than replaying a fixed sample.
def _explorer_error(exc: Exception) -> dict:
    logger.warning("model explorer: %s", exc)
    return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/model-explorer/curves")
async def explorer_curves(distanceKm: float = Query(default=25.0, ge=1.0, le=500.0)):
    """Risk and fuel swept across concentration, for every Polar Class."""
    try:
        from .model_explorer import curves
        return curves(distanceKm)
    except Exception as exc:  # noqa: BLE001
        return _explorer_error(exc)


@app.get("/api/model-explorer/point")
async def explorer_point(
    sic: float = Query(default=0.7, ge=0.0, le=1.0),
    polarClass: str = Query(default="PC4", max_length=16),
    distanceKm: float = Query(default=25.0, ge=1.0, le=500.0),
):
    """One operating point, evaluated by the two deterministic artifacts."""
    try:
        from .model_explorer import point
        return point(sic, polarClass, distanceKm)
    except Exception as exc:  # noqa: BLE001
        return _explorer_error(exc)


@app.get("/api/model-explorer/seaice")
async def explorer_seaice(date: str = Query(default="2018-12-20", max_length=10)):
    """A full 332x316 forecast grid from seaice_forecast.pkl, packed for canvas."""
    try:
        from .model_explorer import seaice_field
        return seaice_field(date)
    except Exception as exc:  # noqa: BLE001
        return _explorer_error(exc)


@app.get("/api/model-explorer/drift")
async def explorer_drift(
    date: str = Query(default="", max_length=10),
    limit: int = Query(default=400, ge=1, le=2000),
):
    """Per-berg 24 h drift vectors from iceberg_drift.pkl."""
    try:
        from .model_explorer import drift_field
        return drift_field(date, limit)
    except Exception as exc:  # noqa: BLE001
        return _explorer_error(exc)


# --------------------------------------------------------------- telemetry
# Live vessel state forwarded by the shipboard gateway (Raspberry Pi) via the
# shore listener. Display only - see app/telemetry.py for why this must not be
# wired into the models.
class TelemetryEnvelope(BaseModel):
    seq: Optional[int] = None
    source: str = ""
    timestamp: str = ""
    priority: int = 0
    payload: dict = Field(default_factory=dict)


@app.post("/api/telemetry/ingest")
async def telemetry_ingest(envelope: TelemetryEnvelope):
    """Accept one record from the shore listener."""
    from .telemetry import ingest
    return ingest(envelope.model_dump())


@app.get("/api/telemetry/live")
async def telemetry_live():
    """Last-known vessel position and instrument readings."""
    from .telemetry import live
    return live()


@app.post("/api/telemetry/reset")
async def telemetry_reset():
    """Clear the state so a demo starts from an empty dashboard."""
    from .telemetry import reset
    return reset()


@app.get("/api/system-status")
async def system_status():
    """One call that answers "is this system actually working right now".

    /api/model-status covers the three model-backed map layers. This covers
    everything a reviewer or an operator needs before trusting a route: every
    component, the date of the environmental data behind it, and whether the
    shipboard telemetry link is up.

    Each component is probed rather than assumed. "live" means it was exercised
    during this request, not that a file exists on disk.
    """
    from datetime import datetime

    components: dict[str, str] = {}
    detail: dict[str, str] = {}

    def probe(name: str, fn) -> None:
        try:
            fn()
            components[name] = "live"
        except Exception as exc:  # noqa: BLE001
            components[name] = "unavailable"
            detail[name] = f"{type(exc).__name__}: {exc}"

    probe("sea_ice", lambda: __import__(
        "app.model_bridge", fromlist=["sic_at_point"]).sic_at_point(-70.0, 0.0))
    probe("iceberg", lambda: __import__(
        "app.model_bridge", fromlist=["_drift_bundle"])._drift_bundle())

    def _polaris():
        from .model_bridge import ensure_seaice_importable
        ensure_seaice_importable()
        from seaice_forecast.risk.polaris import risk_ice
        risk_ice(0.7, polar_class="PC4")
    probe("polaris", _polaris)

    def _fuel():
        from .model_bridge import ensure_seaice_importable
        ensure_seaice_importable()
        from seaice_forecast.fuel import fuel_per_cell_array
        fuel_per_cell_array(25.0, 0.5, polar_class="PC4")
    probe("fuel", _fuel)

    def _routing():
        from .route_engine import ice_data_source
        if "unavailable" in ice_data_source():
            raise RuntimeError("no sea-ice field, so routing would have nothing to cost")
    probe("routing", _routing)

    try:
        from .telemetry import live as telemetry_live
        tel = telemetry_live()
        components["telemetry"] = "connected" if tel.get("connected") else "no_data"
        if tel.get("lastSeen"):
            detail["telemetry"] = f"last record {tel['lastSeen']}"
    except Exception as exc:  # noqa: BLE001
        components["telemetry"] = "unavailable"
        detail["telemetry"] = f"{type(exc).__name__}: {exc}"

    # The newest day in the processed archive. Everything this system forecasts
    # is anchored to it, so it is the single most load-bearing fact here: the
    # system is not real-time and this is the number that proves it.
    env_date = None
    try:
        from .model_bridge import DAILY
        years = sorted(p for p in DAILY.iterdir() if p.is_dir() and p.name.isdigit())
        if years:
            days = sorted(f.stem for f in years[-1].glob("*.npz"))
            if days:
                env_date = datetime.strptime(days[-1], "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        pass

    model_backed = ("sea_ice", "iceberg", "polaris", "fuel", "routing")
    degraded = [k for k in model_backed if components.get(k) != "live"]

    return {
        **components,
        "environment_data_date": env_date,
        "realtime": False,
        "allLive": not degraded,
        "degraded": degraded,
        "detail": detail,
        "note": (
            "environment_data_date is the newest day in the processed archive. "
            "Every forecast is historical reanalysis anchored to it, not a live feed."
        ),
    }


@app.post("/api/route-alternatives")
async def route_alternatives(req: RouteRequest):
    """Plan the same passage under every objective and return the trade-offs.

    A single "optimal" route hides the decision. The objectives genuinely
    disagree - the safe track is longer, the fuel-optimal one spends more time in
    ice than the fastest - and the master is the one who has to weigh that. This
    returns all four so the trade-off is visible rather than resolved silently on
    the reader's behalf.

    No blended "balanced" objective is offered. Blending needs weights, and any
    weight chosen here would be this system's opinion about how many tonnes of
    fuel a unit of ice risk is worth. That is the master's call, and inventing a
    number for it would dress a guess up as an optimum.
    """
    try:
        engine: RouteEngine = app.state.engine
    except AttributeError:
        raise HTTPException(503, "route engine not ready")

    dep = _parse_departure(req.departureTimeUTC)
    objectives = [
        ("distance", "Shortest", "Least distance over ground"),
        ("time", "Fastest", "Least time, using current and wind along track"),
        ("fuel", "Fuel-optimised", "Least burn, pricing ice through speed collapse"),
        ("safety", "Safest", "Least exposure to ice, weather and icebergs"),
    ]

    routes, rows = {}, []
    for key, label, description in objectives:
        try:
            res = engine.compute_route(
                [w for w in req.waypoints],
                speed_knots=req.speedKnots,
                departure_time_utc=dep,
                optimize_for=key,
                vessel_type=req.vesselType,
                draft_meters=req.draftMeters,
                ice_class=req.iceClass,
            )
        except RouteError as exc:
            rows.append({"key": key, "label": label, "description": description,
                         "available": False, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            rows.append({"key": key, "label": label, "description": description,
                         "available": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        routes[key] = res
        rows.append({
            "key": key, "label": label, "description": description, "available": True,
            "distanceNm": res.get("totalDistanceNm"),
            "durationHours": res.get("totalDurationHours"),
            "fuelTonnes": res.get("estimatedFuelTons"),
            "distanceInIceNm": res.get("totalDistanceInSeaIceNm"),
            "maxIceConcentrationPct": res.get("maxSeaIceConcentrationPct"),
            "maxPolarisRisk": res.get("maxPolarisRisk"),
            "meanPolarisRisk": res.get("meanPolarisRisk"),
            "safetyScore": res.get("safetyScore"),
            "closestIcebergNm": (res.get("explanation") or {}).get("closestIcebergNm"),
        })

    available = [r for r in rows if r.get("available")]
    if not available:
        raise HTTPException(422, "no route could be planned under any objective")

    # Express every option as a delta from the shortest route, because "18% more
    # fuel to halve peak ice risk" is a decision a reader can actually make;
    # four absolute numbers are not.
    base = next((r for r in available if r["key"] == "distance"), available[0])
    for r in available:
        def pct(field):
            b, v = base.get(field), r.get(field)
            if not b or v is None:
                return None
            return round((v - b) / b * 100.0, 1)
        r["deltaVsShortest"] = {
            "distancePct": pct("distanceNm"),
            "fuelPct": pct("fuelTonnes"),
            "durationPct": pct("durationHours"),
            "iceExposurePct": pct("distanceInIceNm"),
        }

    def best(field, lowest=True):
        vals = [(r[field], r["key"]) for r in available if r.get(field) is not None]
        if not vals:
            return None
        return (min(vals) if lowest else max(vals))[1]

    # An objective that loses on its own metric is a calibration fault, not a
    # trade-off. Detect it here and say so, rather than shipping a route
    # labelled "Safest" that another option beats on every safety measure.
    # The comparison endpoint exists precisely to make disagreements visible;
    # hiding this one would defeat the point of having it.
    warnings = []
    checks = [
        ("safety", "maxPolarisRisk", "peak POLARIS risk"),
        ("safety", "distanceInIceNm", "distance in ice"),
        ("fuel", "fuelTonnes", "fuel burn"),
        ("time", "durationHours", "duration"),
        ("distance", "distanceNm", "distance"),
    ]
    for key, field, human in checks:
        row = next((r for r in available if r["key"] == key), None)
        if not row or row.get(field) is None:
            continue
        better = [r for r in available
                  if r["key"] != key and r.get(field) is not None
                  and r[field] < row[field]]
        if better:
            winner = min(better, key=lambda r: r[field])
            warnings.append({
                "objective": key,
                "metric": human,
                "detail": (
                    f"the '{key}' objective scores {row[field]} on {human}, "
                    f"worse than '{winner['key']}' at {winner[field]}"
                ),
            })

    return {
        "comparison": rows,
        "best": {
            "shortest": best("distanceNm"),
            "fastest": best("durationHours"),
            "leastFuel": best("fuelTonnes"),
            "leastIceExposure": best("distanceInIceNm"),
            "lowestPeakRisk": best("maxPolarisRisk"),
        },
        "routes": routes,
        "iceDataSource": base.get("explanation", {}).get("iceDataSource")
        if isinstance(base.get("explanation"), dict) else None,
        "warnings": warnings,
        "note": (
            "Objectives genuinely disagree; no blended option is offered because "
            "blending requires weighing fuel against risk, which is the master's "
            "judgement and not this system's."
        ),
        "warningsNote": (
            "A non-empty 'warnings' list means an objective lost on its own "
            "metric, which is a cost-function calibration fault rather than a "
            "trade-off. Treat that option's label as unreliable."
        ),
        "caveat": (
            "All options costed against historical reanalysis. Advisory only."
        ),
    }


class ReplayRequest(BaseModel):
    waypoints: list = Field(default_factory=list)
    startDate: str = "2018-07-15"
    steps: int = Field(default=4, ge=1, le=12)
    stepDays: int = Field(default=2, ge=1, le=30)
    speedKnots: float = 12.0
    vesselType: str = "research"
    draftMeters: float = 7.0
    iceClass: str = "pc5"
    optimizeFor: str = "safety"


@app.post("/api/replay")
async def historical_replay(req: ReplayRequest):
    """Replay archived conditions, re-planning the route at each step.

    Pick a date whose outcome is already known, watch what the models predicted
    and what route followed, and step forward as conditions evolve. This is the
    closest this system can get to demonstrating operation, and in one respect
    it is better than a live demo: the answer is already on the record.
    """
    engine = getattr(app.state, "engine", None)
    try:
        from .replay import replay
        return replay(
            waypoints=[w for w in req.waypoints],
            start_date=req.startDate, steps=req.steps, step_days=req.stepDays,
            speed_knots=req.speedKnots, vessel_type=req.vesselType,
            draft_meters=req.draftMeters, ice_class=req.iceClass,
            optimize_for=req.optimizeFor, engine=engine,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("replay failed")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/uncertainty")
async def uncertainty_summary():
    """Measured confidence for both trained models.

    Derived from held-out error, not assumed. The sea-ice figures show the model
    is near-perfect over open water and worst in the marginal ice zone - least
    confident exactly where a vessel operates, which is worth surfacing rather
    than averaging away.
    """
    try:
        from .uncertainty import summary
        return summary()
    except Exception as exc:  # noqa: BLE001
        logger.warning("uncertainty unavailable: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/uncertainty/at")
async def uncertainty_at(
    sic: float = Query(..., ge=0.0, le=1.0),
    gradient: float = Query(default=None, ge=0.0, le=2.0),
):
    """Expected error for a sea-ice prediction at this concentration/gradient."""
    try:
        from .uncertainty import sea_ice_uncertainty
        return sea_ice_uncertainty(sic, gradient)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------- bridge console
# Shapes the local-dashboard/ console expects, served from the telemetry
# gateway and the trained models. Its original backend generated these from
# random(); that backend was dropped rather than imported.
@app.get("/api/vessel")
async def console_vessel():
    from .bridge_console import vessel
    return vessel()


@app.get("/api/contacts/ais")
async def console_ais():
    from .bridge_console import contacts_ais
    return contacts_ais()


@app.get("/api/contacts/radar")
async def console_radar():
    """No radar source exists; this reports that rather than inventing contacts."""
    from .bridge_console import contacts_radar
    return contacts_radar()


@app.get("/api/environment")
async def console_environment():
    from .bridge_console import environment
    return environment()


@app.get("/api/dashboard")
async def console_dashboard():
    from .bridge_console import dashboard
    return dashboard()
