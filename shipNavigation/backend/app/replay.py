"""Historical replay: step through real past conditions and re-plan at each step.

The archive ends 2018-12-31, so this system cannot demonstrate live operation.
It can do something a live demo cannot: take a date whose outcome is already
known, show what the models predicted, show the route that followed, and step
forward to watch both change as conditions evolve.

That also exercises the capability an operational system needs and a one-shot
planner lacks - re-planning. At each step the route is recomputed against that
day's ice, and compared with the route from the step before. A track that never
changes as the ice moves is not decision support, it is a drawing.

Each step reports:
  * the ice field the models saw that day
  * where the drift model put the icebergs
  * the route planned against those conditions
  * how that route differs from the previous step's, and whether the change
    is large enough to be worth a watch officer's attention
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

# Below this, a re-plan is noise: grid snapping and smoothing move a track by
# small amounts run to run, and flagging that as "route changed" would train a
# reader to ignore the flag.
MATERIAL_DISTANCE_PCT = 2.0
MATERIAL_RISK_DELTA = 0.05


def _ice_summary(date_str: str) -> dict[str, Any]:
    """What the forecaster saw on this date."""
    from .model_bridge import MASK_P, predict_sic

    grid = predict_sic(date_str)
    mask = np.load(MASK_P)
    ocean = grid[mask != 0]
    iced = ocean[ocean >= 0.15]
    cell_km2 = 25.0 * 25.0
    return {
        "meanConcentration": round(float(ocean.mean()), 4),
        "maxConcentration": round(float(grid.max()), 4),
        "iceExtentMkm2": round(float(iced.size) * cell_km2 / 1e6, 3),
        "iceAreaMkm2": round(float(iced.sum()) * cell_km2 / 1e6, 3),
    }


def _icebergs_on(date_str: str, limit: int = 40) -> dict[str, Any]:
    """Drift-model output for the bergs observed nearest this date."""
    from .model_explorer import _artifact, _drift_frame

    model = _artifact("iceberg_drift")
    frame = _drift_frame().dropna(subset=model.features)
    target = datetime.strptime(date_str, "%Y-%m-%d")

    # The NIC fixes are irregular, so take the nearest observation day rather
    # than requiring an exact match that usually will not exist.
    import pandas as pd

    days = pd.Series(frame["date"].dt.normalize().unique())
    if days.empty:
        return {"count": 0, "movingCount": 0, "bergs": []}
    # Nearest observation day. The NIC fixes are irregular, so an exact match
    # usually does not exist and requiring one would silently return nothing.
    nearest = days.iloc[(days - pd.Timestamp(target)).abs().argmin()]
    sel = frame[frame["date"].dt.normalize() == nearest].head(limit)
    if sel.empty:
        return {"count": 0, "movingCount": 0, "bergs": []}

    r = model.predict(sel)
    bergs = [
        {
            "bergId": str(row.berg_id),
            "lat": round(float(row.lat), 4),
            "lon": round(float(row.lon), 4),
            "moving": bool(r["moving"][i]),
            "driftKm24h": round(float(r["drift_km_24h"][i]), 2),
        }
        for i, row in enumerate(sel.itertuples())
    ]
    return {
        "observedOn": nearest.strftime("%Y-%m-%d"),
        "count": len(bergs),
        "movingCount": sum(b["moving"] for b in bergs),
        "bergs": bergs,
    }


def _compare(previous: dict | None, current: dict) -> dict[str, Any]:
    """Did conditions move the route enough to matter?"""
    if previous is None:
        return {"replanned": False, "material": False,
                "note": "initial plan"}

    d_prev = previous.get("distanceNm") or 0.0
    d_cur = current.get("distanceNm") or 0.0
    r_prev = previous.get("maxPolarisRisk") or 0.0
    r_cur = current.get("maxPolarisRisk") or 0.0

    dist_pct = ((d_cur - d_prev) / d_prev * 100.0) if d_prev else 0.0
    risk_delta = r_cur - r_prev
    material = (abs(dist_pct) >= MATERIAL_DISTANCE_PCT
                or abs(risk_delta) >= MATERIAL_RISK_DELTA)

    if not material:
        note = "conditions changed but the recommended track did not materially move"
    elif risk_delta > 0:
        note = (f"ice risk rose {risk_delta:+.2f}; the re-planned track is "
                f"{dist_pct:+.1f}% in distance")
    else:
        note = (f"ice risk fell {risk_delta:+.2f}; a {dist_pct:+.1f}% distance "
                f"change is now available")

    return {
        "replanned": True,
        "material": material,
        "distanceChangePct": round(dist_pct, 2),
        "riskChange": round(risk_delta, 4),
        "note": note,
    }


def replay(waypoints: list, start_date: str, steps: int = 4, step_days: int = 2,
           speed_knots: float = 12.0, vessel_type: str = "research",
           draft_meters: float = 7.0, ice_class: str = "pc5",
           optimize_for: str = "safety", engine=None) -> dict[str, Any]:
    """Step through history, re-planning at each step."""
    from .route_engine import RouteError

    start = datetime.strptime(start_date, "%Y-%m-%d")
    frames: list[dict[str, Any]] = []
    previous: dict | None = None

    for i in range(steps):
        when = start + timedelta(days=i * step_days)
        date_str = when.strftime("%Y-%m-%d")

        frame: dict[str, Any] = {"step": i, "date": date_str,
                                 "hoursFromStart": i * step_days * 24}
        try:
            frame["ice"] = _ice_summary(date_str)
        except Exception as exc:  # noqa: BLE001
            frame["ice"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            frame["icebergs"] = _icebergs_on(date_str)
        except Exception as exc:  # noqa: BLE001
            frame["icebergs"] = {"error": f"{type(exc).__name__}: {exc}"}

        route_summary: dict | None = None
        if engine is not None:
            try:
                res = engine.compute_route(
                    waypoints, speed_knots=speed_knots, departure_time_utc=when,
                    optimize_for=optimize_for, vessel_type=vessel_type,
                    draft_meters=draft_meters, ice_class=ice_class,
                )
                route_summary = {
                    "distanceNm": res.get("totalDistanceNm"),
                    "durationHours": res.get("totalDurationHours"),
                    "fuelTonnes": res.get("estimatedFuelTons"),
                    "distanceInIceNm": res.get("totalDistanceInSeaIceNm"),
                    "maxPolarisRisk": res.get("maxPolarisRisk"),
                    "path": [p for leg in res.get("legs", []) for p in leg.get("path", [])],
                }
                frame["route"] = {k: v for k, v in route_summary.items() if k != "path"}
                frame["path"] = route_summary["path"]
            except RouteError as exc:
                frame["route"] = {"error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                frame["route"] = {"error": f"{type(exc).__name__}: {exc}"}

        frame["change"] = _compare(previous, route_summary or {})
        if route_summary:
            previous = route_summary
        frames.append(frame)

    replans = [f for f in frames if f["change"].get("material")]
    return {
        "startDate": start_date,
        "steps": steps,
        "stepDays": step_days,
        "optimizeFor": optimize_for,
        "frames": frames,
        "materialReplans": len(replans),
        "summary": (
            f"{len(frames)} steps from {start_date} at {step_days}-day intervals; "
            f"{len(replans)} produced a materially different recommendation."
        ),
        "note": (
            "Replay of real archived conditions. The outcome of these dates is "
            "already known, which is what makes this a test of the system's "
            "judgement rather than a demonstration of its confidence."
        ),
        "caveat": (
            "Historical reanalysis, not live operation. A material re-plan "
            "threshold of 2% distance or 0.05 risk keeps grid noise from being "
            "reported as a route change."
        ),
    }
