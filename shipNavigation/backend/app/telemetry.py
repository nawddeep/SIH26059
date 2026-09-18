"""Last-known vessel state, fed by the shipboard telemetry gateway.

The gateway (a Raspberry Pi aboard the vessel) pulls GPS, AIS, weather and
engine data off the ship's instruments, buffers it, and forwards it to the
shore listener over a constrained satellite link. The shore listener hands each
record here.

This is deliberately a display surface and nothing more. It does not feed the
forecast, drift or routing models, and it must not be described as doing so:
the telemetry is current, while the processed ice archive ends 2018-12-31, so
any "live risk score" built on top of this would be pairing a real position
with an eight-year-old environment. Showing where the ship is and what its
instruments report is honest; inferring ice risk from it would not be.

State is in memory on purpose. It is last-known-value only - no history, no
persistence - because the durable copy already exists in the gateway's SQLite
queue and in the shore listener's log. Restarting the API should not pretend to
know anything the ship has not sent since.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

# One lock around the whole state: writes arrive from the shore listener on one
# thread while the API reads on another, and the state is small enough that
# finer-grained locking would be complexity without benefit.
_lock = threading.Lock()

_state: dict[str, Any] = {
    "gps": None,        # last position fix
    "ais": None,        # last own-ship AIS report
    "weather": None,    # last weather observation
    "engine": None,     # last engine reading
}
_meta: dict[str, Any] = {
    "recordsReceived": 0,
    "bySource": {},
    "firstSeen": None,
    "lastSeen": None,
    "alerts": [],       # most recent alerts, newest first
}

MAX_ALERTS = 8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ingest(envelope: dict) -> dict:
    """Record one envelope from the gateway. Returns a short acknowledgement.

    Envelope shape matches what sender.py transmits:
        {seq, source, timestamp, priority, payload}
    """
    source = str(envelope.get("source", "")).lower()
    payload = envelope.get("payload") or {}
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "payload must be an object"}

    with _lock:
        stamp = _now()
        _meta["recordsReceived"] += 1
        _meta["bySource"][source] = _meta["bySource"].get(source, 0) + 1
        if _meta["firstSeen"] is None:
            _meta["firstSeen"] = stamp
        _meta["lastSeen"] = stamp

        if source == "gps":
            # Only RMC carries speed and course; GGA is fix-quality detail and
            # would otherwise overwrite the good fix with a partial one.
            if payload.get("type") == "position_fix":
                _state["gps"] = {**payload, "receivedAt": stamp, "seq": envelope.get("seq")}
        elif source == "ais":
            if payload.get("own_ship"):
                _state["ais"] = {**payload, "receivedAt": stamp, "seq": envelope.get("seq")}
        elif source in ("weather", "engine"):
            _state[source] = {**payload, "receivedAt": stamp, "seq": envelope.get("seq")}

        for alert in payload.get("alerts") or []:
            _meta["alerts"].insert(0, {"source": source, "alert": alert, "at": stamp})
        del _meta["alerts"][MAX_ALERTS:]

        return {"ok": True, "recordsReceived": _meta["recordsReceived"]}


def live() -> dict:
    """Current vessel state for the map and the telemetry strip."""
    with _lock:
        gps = _state["gps"]
        ais = _state["ais"]

        # Prefer the GPS fix for position; fall back to AIS if GPS has not
        # arrived yet. Both come from the same receiver aboard, so they agree.
        position = None
        if gps and gps.get("lat") is not None:
            position = {
                "lat": gps["lat"], "lon": gps["lon"],
                "speedKn": gps.get("speed_kn"), "courseDeg": gps.get("course_deg"),
                "from": "gps", "receivedAt": gps["receivedAt"],
            }
        elif ais and ais.get("lat") is not None:
            position = {
                "lat": ais["lat"], "lon": ais["lon"],
                "speedKn": ais.get("speed"), "courseDeg": ais.get("course"),
                "from": "ais", "receivedAt": ais["receivedAt"],
            }

        return {
            "connected": _meta["lastSeen"] is not None,
            "vesselName": (ais or {}).get("name"),
            "mmsi": (ais or {}).get("mmsi"),
            "position": position,
            "weather": _state["weather"],
            "engine": _state["engine"],
            "alerts": list(_meta["alerts"]),
            "recordsReceived": _meta["recordsReceived"],
            "bySource": dict(_meta["bySource"]),
            "firstSeen": _meta["firstSeen"],
            "lastSeen": _meta["lastSeen"],
            "note": (
                "Position and instrument readings forwarded by the shipboard "
                "gateway. Display only - not an input to the forecast, drift or "
                "routing models."
            ),
        }


def reset() -> dict:
    """Clear state, so a demo can start from a known-empty dashboard."""
    with _lock:
        for key in _state:
            _state[key] = None
        _meta.update({"recordsReceived": 0, "bySource": {},
                      "firstSeen": None, "lastSeen": None, "alerts": []})
    return {"ok": True}
