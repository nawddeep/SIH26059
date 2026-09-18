"""Interactive endpoints behind the model dashboard's explorer panel.

Every number returned here comes from a `.pkl` in model_exports/, loaded from
disk with joblib - not from the model objects the map endpoints keep in memory.
That is the point of the panel: the reader changes an input, and the response is
the exported artifact's own `predict()` output for it.

The four artifacts answer different shapes of question, so each gets its own
function rather than one generic dispatcher:

  curves()       sweep concentration across every Polar Class   (risk + fuel)
  point()        one operating point, for the readout            (risk + fuel)
  seaice_field() a 332x316 forecast grid, packed for the canvas  (sea ice)
  drift_field()  per-berg 24 h displacement vectors              (drift)
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .model_bridge import DAILY, MODEL_ROOT, ensure_seaice_importable

EXPORT_DIR = MODEL_ROOT / "model_exports"

# Both artifacts accept these; the fuel model also knows the Baltic classes, but
# a selector is only useful if every chart on the page can answer for the choice.
POLAR_CLASSES = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "UNCLASSED"]

SIC_STEPS = 61          # 0.000, 0.0167, ... 1.000 - smooth at chart width
LAND = 255              # sentinel in the packed grid; SIC quantises to 0..254


@lru_cache(maxsize=8)
def _artifact(key: str):
    """joblib.load one exported artifact, cached for the process lifetime.

    The sea-ice artifact rebuilds its architecture from the seaice_forecast
    package on first predict(), so that has to be importable before it runs -
    the same way an sklearn pickle needs sklearn.
    """
    import joblib

    ensure_seaice_importable()
    path = EXPORT_DIR / f"{key}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Build it with "
            f"backend/scripts/export_model_pickles.py"
        )
    return joblib.load(path)


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _clean_class(polar_class: str) -> str:
    k = str(polar_class or "").strip().upper()
    return k if k in POLAR_CLASSES else "PC4"


# ------------------------------------------------------------------- curves
def curves(distance_km: float = 25.0) -> dict[str, Any]:
    """Risk, fuel, speed and transit across the full concentration range.

    Returned for every Polar Class at once so the chart can draw the selected
    hull against the rest of the ladder without a round trip per selection.
    """
    risk_model = _artifact("polaris_risk")
    fuel_model = _artifact("fuel_consumption")
    distance_km = float(max(1.0, min(500.0, distance_km)))

    sic = np.linspace(0.0, 1.0, SIC_STEPS)
    series: dict[str, Any] = {}
    for pc in POLAR_CLASSES:
        risk = np.asarray(risk_model.predict(sic, pc), dtype=float)
        fuel = fuel_model.predict(sic, pc, distance_km)
        series[pc] = {
            "risk": [round(float(v), 4) for v in risk],
            "fuelTonnes": [round(float(v), 4) for v in np.asarray(fuel["fuel_tonnes"])],
            "speedKnots": [round(float(v), 3) for v in np.asarray(fuel["speed_knots"])],
            "transitHours": [round(float(v), 3) for v in np.asarray(fuel["transit_hours"])],
        }

    return {
        "sic": [round(float(v), 4) for v in sic],
        "polarClasses": POLAR_CLASSES,
        "distanceKm": distance_km,
        "series": series,
        "source": "polaris_risk.pkl + fuel_consumption.pkl",
    }


# -------------------------------------------------------------------- point
def point(sic: float, polar_class: str = "PC4", distance_km: float = 25.0) -> dict[str, Any]:
    """One operating point, evaluated by the artifacts rather than interpolated.

    The chart could read this off the swept curve, but a value the reader typed
    deserves the model's answer for exactly that value.
    """
    pc = _clean_class(polar_class)
    s = _clamp01(sic)
    distance_km = float(max(1.0, min(500.0, distance_km)))

    risk = float(_artifact("polaris_risk").predict(s, pc))
    fuel = _artifact("fuel_consumption").predict(s, pc, distance_km)

    open_water = _artifact("fuel_consumption").predict(0.0, pc, distance_km)
    burn = float(fuel["fuel_tonnes"])
    baseline = float(open_water["fuel_tonnes"])

    return {
        "sic": round(s, 4),
        "polarClass": pc,
        "distanceKm": distance_km,
        "risk": round(risk, 4),
        "fuelTonnes": round(burn, 4),
        "speedKnots": round(float(fuel["speed_knots"]), 3),
        "transitHours": round(float(fuel["transit_hours"]), 3),
        "openWaterFuelTonnes": round(baseline, 4),
        "fuelPenaltyX": round(burn / baseline, 3) if baseline > 0 else None,
        "source": "polaris_risk.pkl + fuel_consumption.pkl",
    }


# --------------------------------------------------------------- sea ice
def _history_for(date: datetime) -> np.ndarray:
    """The seven daily fields ending on `date`, as the artifact expects them."""
    days = []
    for i in range(7):
        d = (date - timedelta(days=6 - i)).date()
        f = DAILY / str(d.year) / f"{d.strftime('%Y%m%d')}.npz"
        if not f.exists():
            raise FileNotFoundError(f"no regridded day for {d} - archive covers 2008-01-01 to 2018-12-31")
        days.append(np.load(f)["data"].astype(np.float32))
    return np.stack(days)


@lru_cache(maxsize=8)
def seaice_field(date_str: str) -> dict[str, Any]:
    """Run seaice_forecast.pkl for one date and pack the grid for a canvas.

    104,912 cells is far too many SVG rects, and as JSON floats it is a ~700 KB
    payload for a picture. Quantising to one byte per cell and base64-ing it is
    ~140 KB and loses nothing a heatmap could show. Land carries a sentinel so
    the client can paint it differently from genuinely ice-free ocean.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    model = _artifact("seaice_forecast")

    grid = np.asarray(model.predict(_history_for(date)), dtype=np.float32)
    mask = np.asarray(model.mask)

    packed = np.rint(np.clip(grid, 0.0, 1.0) * 254.0).astype(np.uint8)
    packed[mask == 0] = LAND

    ocean = grid[mask != 0]
    iced = ocean[ocean >= 0.15]
    cell_km2 = 25.0 * 25.0

    return {
        "date": date_str,
        "height": int(grid.shape[0]),
        "width": int(grid.shape[1]),
        "land": LAND,
        "scale": 254.0,
        "grid": base64.b64encode(packed.tobytes()).decode("ascii"),
        "stats": {
            "meanOcean": round(float(ocean.mean()), 4),
            "max": round(float(grid.max()), 4),
            "iceCells": int(iced.size),
            "oceanCells": int(ocean.size),
            "iceAreaMkm2": round(float(iced.sum()) * cell_km2 / 1e6, 3),
            "extentMkm2": round(float(iced.size) * cell_km2 / 1e6, 3),
        },
        "source": "seaice_forecast.pkl",
        "note": "Historical reanalysis input, not a live feed - the archive ends 2018-12-31.",
    }


# ----------------------------------------------------------------- drift
@lru_cache(maxsize=1)
def _drift_frame():
    """The berg fixes, with the derived columns the artifact's features expect."""
    import pandas as pd

    from .model_bridge import DRIFT_CSV

    d = pd.read_csv(DRIFT_CSV)
    d["date"] = pd.to_datetime(d["date"])
    mo = d["date"].dt.month
    d["month_s"] = np.sin(2 * np.pi * mo / 12)
    d["month_c"] = np.cos(2 * np.pi * mo / 12)
    d["wind_spd"] = np.hypot(d.u_wind, d.v_wind)
    d["curr_spd"] = np.hypot(d.u_curr, d.v_curr)
    return d


def drift_dates(limit: int = 400) -> list[str]:
    """Dates with enough simultaneous fixes to make a vector field worth drawing."""
    d = _drift_frame()
    counts = d.groupby(d["date"].dt.strftime("%Y-%m-%d")).size()
    return [str(k) for k in counts[counts >= 8].index[-limit:]]


def drift_field(date_str: str = "", limit: int = 400) -> dict[str, Any]:
    """Per-berg 24 h displacement from iceberg_drift.pkl, as plottable vectors.

    Stationary bergs are returned too, flagged rather than filtered: the first
    stage of this model is a moving/stationary gate, and a field that quietly
    dropped everything it gated off would hide half of what it does.
    """
    model = _artifact("iceberg_drift")
    d = _drift_frame().dropna(subset=model.features)

    if date_str:
        sel = d[d["date"].dt.strftime("%Y-%m-%d") == date_str]
        if sel.empty:
            raise ValueError(f"no berg fixes on {date_str}")
    else:
        # Default to the most recent day that actually has a crowd on it.
        by_day = d.groupby(d["date"].dt.strftime("%Y-%m-%d"))
        best = [k for k, g in by_day if len(g) >= 8]
        date_str = best[-1] if best else d["date"].dt.strftime("%Y-%m-%d").iloc[-1]
        sel = d[d["date"].dt.strftime("%Y-%m-%d") == date_str]

    sel = sel.head(int(max(1, min(2000, limit))))
    r = model.predict(sel)

    bergs = []
    for i, (_, row) in enumerate(sel.iterrows()):
        u, v = float(r["u_ms"][i]), float(r["v_ms"][i])
        bergs.append({
            "bergId": str(row["berg_id"]),
            "lat": round(float(row["lat"]), 4),
            "lon": round(float(row["lon"]), 4),
            "moving": bool(r["moving"][i]),
            "uMs": round(u, 5),
            "vMs": round(v, 5),
            "driftKm24h": round(float(r["drift_km_24h"][i]), 3),
            "bearingDeg": round(float((np.degrees(np.arctan2(u, v)) + 360.0) % 360.0), 1),
            "speedKnots": round(float(np.hypot(u, v)) * 1.94384, 4),
            "sizeNm": round(float(row["size_1"]), 1),
        })

    moving = [b for b in bergs if b["moving"]]
    return {
        "date": date_str,
        "count": len(bergs),
        "movingCount": len(moving),
        "maxDriftKm24h": round(max((b["driftKm24h"] for b in bergs), default=0.0), 3),
        "meanMovingDriftKm24h": (
            round(sum(b["driftKm24h"] for b in moving) / len(moving), 3) if moving else 0.0
        ),
        "bergs": bergs,
        "availableDates": drift_dates(),
        "source": "iceberg_drift.pkl",
    }
