"""Endpoints for the bridge-console dashboard, backed by real sources.

The dashboard in local-dashboard/ shipped with its own backend that generated
vessel speed, wind and contact bearings from `random.random()`. That backend was
dropped rather than imported. These endpoints serve the same shapes from the
real telemetry gateway and the trained models instead.

Where no real source exists the endpoint says so rather than inventing one. That
applies to radar: this system has no radar feed, so /api/contacts/radar returns
an empty list with a reason. A console that draws plausible radar contacts out of
a random number generator is worse than one that shows none, because the officer
cannot tell the difference.
"""
from __future__ import annotations

from typing import Any


def _telemetry() -> dict[str, Any]:
    from .telemetry import live
    return live()


def vessel() -> dict[str, Any]:
    """Own-ship state, from the telemetry gateway."""
    t = _telemetry()
    pos = t.get("position") or {}
    engine = t.get("engine") or {}
    return {
        "name": t.get("vesselName"),
        "mmsi": t.get("mmsi"),
        "connected": t.get("connected", False),
        "latitude": pos.get("lat"),
        "longitude": pos.get("lon"),
        "speedKnots": pos.get("speedKn"),
        "headingDeg": pos.get("courseDeg"),
        "rpm": engine.get("rpm"),
        "fuelRemainingPct": engine.get("fuel_remaining_pct"),
        "engineTempC": engine.get("engine_temp_c"),
        "lastUpdate": t.get("lastSeen"),
        "source": "shipboard telemetry gateway",
        "note": None if t.get("connected") else
                "no telemetry received - start the gateway with --forward",
    }


def contacts_ais() -> dict[str, Any]:
    """AIS contacts. Only own ship is currently forwarded by the gateway."""
    t = _telemetry()
    pos = t.get("position") or {}
    contacts = []
    if t.get("connected") and pos.get("lat") is not None:
        contacts.append({
            "mmsi": t.get("mmsi"),
            "name": t.get("vesselName"),
            "latitude": pos.get("lat"),
            "longitude": pos.get("lon"),
            "speedKnots": pos.get("speedKn"),
            "headingDeg": pos.get("courseDeg"),
            "ownShip": True,
        })
    return {
        "contacts": contacts,
        "count": len(contacts),
        "source": "shipboard telemetry gateway (AIS feed)",
        "note": (
            "Only own-ship AIS is retained. The gateway forwards nearby contacts "
            "too, but this system keeps last-known-value per source rather than a "
            "contact table, so other vessels are not tracked here."
        ),
    }


def contacts_radar() -> dict[str, Any]:
    """No radar source exists. Says so rather than inventing contacts."""
    return {
        "contacts": [],
        "count": 0,
        "source": "unavailable",
        "note": (
            "This system has no radar feed. The dashboard's original backend "
            "generated radar contacts with random(); that is not reproduced here. "
            "A console showing invented contacts is worse than one showing none, "
            "because the officer cannot tell which is which."
        ),
    }


def environment() -> dict[str, Any]:
    """Conditions: measured where the ship reports them, modelled where it does not."""
    t = _telemetry()
    weather = t.get("weather") or {}

    ice: dict[str, Any] = {"available": False}
    try:
        from .model_bridge import DEFAULT_FORECAST_DATE, sic_at_point
        pos = t.get("position") or {}
        if pos.get("lat") is not None:
            ice = {
                "available": True,
                "concentrationPct": round(float(sic_at_point(pos["lat"], pos["lon"])), 1),
                "atVesselPosition": True,
                "forecastDate": DEFAULT_FORECAST_DATE.strftime("%Y-%m-%d"),
            }
    except Exception as exc:  # noqa: BLE001
        ice = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "windSpeedKn": weather.get("wind_speed_kn"),
        "windDirDeg": weather.get("wind_dir_deg"),
        "airTempC": weather.get("air_temp_c"),
        "seaTempC": weather.get("sea_temp_c"),
        "pressureHpa": weather.get("pressure_hpa"),
        "visibilityKm": weather.get("visibility_km"),
        "alerts": weather.get("alerts", []),
        "seaIce": ice,
        "sources": {
            "weather": "shipboard weather station via the telemetry gateway",
            "seaIce": "U-Net+ConvLSTM forecast (NSIDC CDR v6)",
        },
        "caveat": (
            "Weather is current if the gateway is running. Sea ice is historical "
            "reanalysis - the processed archive ends 2018-12-31 - so the two are "
            "not contemporaneous and must not be read as one picture."
        ),
    }


def dashboard() -> dict[str, Any]:
    """Everything the console shows, in one call."""
    t = _telemetry()
    status: dict[str, Any] = {}
    try:
        from .model_bridge import model_status
        status = model_status()
    except Exception:  # noqa: BLE001
        status = {"allLive": False}

    return {
        "vessel": vessel(),
        "environment": environment(),
        "ais": contacts_ais(),
        "radar": contacts_radar(),
        "models": {
            "allLive": status.get("allLive", False),
            "live": status.get("liveComponents", []),
        },
        "telemetryConnected": t.get("connected", False),
        "recordsReceived": t.get("recordsReceived", 0),
        "realtime": False,
        "note": (
            "Assembled from the telemetry gateway and the trained models. No "
            "value on this console is generated; anything without a real source "
            "is reported as unavailable."
        ),
    }
