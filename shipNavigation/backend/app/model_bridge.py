"""
Bridge to the trained Antarctic models (SIH PS 26059).

Serves sea-ice and iceberg data from the trained models. The simulated
generators this once fell back to have been removed:

  * sea ice   - U-Net + ConvLSTM on NSIDC CDR v6, 332x316 EPSG:3412 grid.
                Validated against persistence and climatology on a held-out
                2017-2018 split (matches persistence at +5d, beats it at +7d).
  * icebergs  - two-stage HistGradientBoosting drift model trained on real NIC
                fixes. 3.96 km RMS 24 h position error, 96.7% within 10 km.

If the model repo or its artifacts are missing, every function here raises and
the caller returns an empty field tagged "unavailable". The map keeps rendering
but shows nothing where the data would be, which is the honest outcome: a chart
that invents ice is more dangerous than one that admits it has none.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# The models live in the parent repository this app is checked out inside:
# <model root>/shipNavigation/backend/app/model_bridge.py -> parents[3].
# ICEBERG_MODEL_ROOT overrides it when the two are checked out separately.
import os

MODEL_ROOT = Path(
    os.environ.get("ICEBERG_MODEL_ROOT") or Path(__file__).resolve().parents[3]
)
SEAICE = MODEL_ROOT / "seaice_forecast"
DRIFT = MODEL_ROOT / "iceberg-drift"
SEAICE_SRC = SEAICE / "src"


def ensure_seaice_importable() -> None:
    """Put the seaice_forecast package on sys.path (risk/fuel/routing modules)."""
    if str(SEAICE_SRC) not in sys.path:
        sys.path.insert(0, str(SEAICE_SRC))

DAILY = SEAICE / "data/processed/regridded/daily"
MASK_P = SEAICE / "data/processed/land_ocean_mask_ps25.npy"
NORM_P = SEAICE / "data/processed/normalization_stats.json"
LONLAT_P = SEAICE / "data/processed/nsidc_ps25_lon_lat.npz"
SIC_CKPT = SEAICE / "models/checkpoints/phase3/sic_unet_convlstm_v001_best.pt"
DRIFT_PKL = DRIFT / "models/drift_twostage.pkl"
DRIFT_CSV = DRIFT / "data/drift_trainable_rich.csv"

VARS = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]

# Latest date with a full 7-day history in the processed archive.
DEFAULT_FORECAST_DATE = datetime(2018, 12, 20)


def models_available() -> bool:
    return all(p.exists() for p in (DAILY, MASK_P, NORM_P, LONLAT_P, SIC_CKPT, DRIFT_PKL, DRIFT_CSV))


# --------------------------------------------------------------------- sea ice
REQUIRED_NORM_VARS = ("wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v")


@lru_cache(maxsize=1)
def _load_norm_stats() -> dict:
    """Load per-variable normalisation stats, refusing anything malformed.

    This file has been silently overwritten before with a single scalar
    {"mean": ..., "std": ...} pair. When that happens every lookup like
    `norm["air_temp"]` misses, normalisation quietly no-ops, the network is fed
    raw Kelvin (~273 against a 0-1 target) and its output saturates to ~1.0
    everywhere. The map then shows a fully ice-covered ocean that looks
    plausible but is entirely wrong - the worst kind of failure.

    So: validate the shape, and fall back to the reference copy rather than
    proceeding with stats that cannot be right.
    """
    def _valid(d):
        return isinstance(d, dict) and all(
            isinstance(d.get(v), dict) and "mean" in d[v] and "std" in d[v]
            for v in REQUIRED_NORM_VARS
        )

    stats = json.loads(NORM_P.read_text())
    if _valid(stats):
        return stats

    ref = NORM_P.with_name("normalization_stats.reference.json")
    if ref.exists():
        backup = json.loads(ref.read_text())
        if _valid(backup):
            import logging
            logging.getLogger(__name__).error(
                "%s is malformed (keys=%s); using reference copy instead.",
                NORM_P.name, sorted(stats)[:6],
            )
            return backup

    raise RuntimeError(
        f"{NORM_P} is malformed and no valid reference copy exists. "
        f"Expected per-variable entries for {REQUIRED_NORM_VARS}, got keys "
        f"{sorted(stats)[:8]}. Regenerate with scripts/data/build_daily_npz.py."
    )


@lru_cache(maxsize=1)
def _sic_model():
    ensure_seaice_importable()
    import torch
    from seaice_forecast.models.unet_convlstm import UNetConvLSTM

    m = UNetConvLSTM(
        in_channels=7, output_channels=1, seq_len=7,
        encoder_channels=[16, 32, 64, 128], convlstm_layers=1,
        dropout=0.0, output_activation="sigmoid",
    )
    ck = torch.load(SIC_CKPT, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, torch


@lru_cache(maxsize=4)
def predict_sic(date_str: str) -> np.ndarray:
    """Run the trained forecaster for `date_str`; returns SIC[332,316] in [0,1]."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    model, torch = _sic_model()
    mask = np.load(MASK_P)
    norm = _load_norm_stats()

    days = []
    for i in range(7):
        d = (date - timedelta(days=6 - i)).date()
        f = DAILY / str(d.year) / f"{d.strftime('%Y%m%d')}.npz"
        if not f.exists():
            raise FileNotFoundError(f"missing regridded day {d}")
        days.append(np.load(f)["data"].astype(np.float32))
    x = np.stack(days)

    for c, v in enumerate(VARS):                 # same normalisation as training
        if v != "sic" and v in norm and norm[v]["std"] > 1e-6:
            x[:, c] = (x[:, c] - norm[v]["mean"]) / norm[v]["std"]
    x[:, 1:, mask == 0] = 0.0

    with torch.no_grad():
        out = model(torch.from_numpy(x).unsqueeze(0).float()).squeeze().numpy()
    out = np.clip(out, 0.0, 1.0)
    out[mask == 0] = 0.0
    return out


@lru_cache(maxsize=1)
def _grid_lookup():
    """Map the frontend's lat/lon sample grid onto nearest PS-25 cells (once)."""
    z = np.load(LONLAT_P)
    glon, glat = z["lons"], z["lats"]
    lats = np.arange(-85.0, -55.0, 1.5)
    lons = np.arange(-180.0, 180.0, 3.0)

    # Compare in 3-D so the +/-180 seam and pole converge correctly.
    gl_r, gt_r = np.radians(glon.ravel()), np.radians(glat.ravel())
    gx = np.cos(gt_r) * np.cos(gl_r); gy = np.cos(gt_r) * np.sin(gl_r); gz = np.sin(gt_r)
    G = np.stack([gx, gy, gz], 1)

    idx = {}
    for la in lats:
        t = np.radians(la)
        for lo in lons:
            p = np.radians(lo)
            v = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)])
            k = int(np.argmax(G @ v))
            idx[(float(la), float(lo))] = (k // glon.shape[1], k % glon.shape[1])
    return lats, lons, idx


def sea_ice_geojson_real(date_str: str | None = None) -> dict:
    """Sea-ice concentration as GeoJSON, driven by the trained forecaster."""
    ds = date_str or DEFAULT_FORECAST_DATE.strftime("%Y-%m-%d")
    sic = predict_sic(ds)
    lats, lons, idx = _grid_lookup()

    feats = []
    for la in lats:
        for lo in lons:
            r, c = idx[(float(la), float(lo))]
            conc = float(sic[r, c]) * 100.0
            if conc < 10.0:
                continue
            hlat, hlon = 0.75, 1.5
            feats.append({
                "type": "Feature",
                "properties": {
                    "concentration": conc,
                    "category": "heavy" if conc >= 60 else "medium" if conc >= 30 else "marginal",
                    "source": "UNet+ConvLSTM forecast (NSIDC CDR v6)",
                    "valid_date": ds,
                },
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lo - hlon, la - hlat], [lo + hlon, la - hlat],
                    [lo + hlon, la + hlat], [lo - hlon, la + hlat],
                    [lo - hlon, la - hlat],
                ]]},
            })
    return {"type": "FeatureCollection", "features": feats}


# -------------------------------------------------------------------- icebergs
@lru_cache(maxsize=1)
def _drift_bundle():
    import joblib
    import pandas as pd

    bundle = joblib.load(DRIFT_PKL)
    d = pd.read_csv(DRIFT_CSV)
    d["date"] = pd.to_datetime(d["date"])
    mo = d["date"].dt.month
    d["month_s"] = np.sin(2 * np.pi * mo / 12)
    d["month_c"] = np.cos(2 * np.pi * mo / 12)
    d["wind_spd"] = np.hypot(d.u_wind, d.v_wind)
    d["curr_spd"] = np.hypot(d.u_curr, d.v_curr)
    return bundle, d


def _size_category(size_1: float) -> str:
    if size_1 >= 30: return "large"
    if size_1 >= 10: return "medium"
    return "small"


def icebergs_with_trajectories_real(
    start_time: datetime, end_time: datetime, step_hours: float = 1.0,
    max_bergs: int = 30,
) -> list[dict[str, Any]]:
    """Real NIC bergs advected by the trained drift model."""
    bundle, d = _drift_bundle()
    feats = bundle["features"]

    anchor = d["date"].max()                      # archive ends 2018; map dates are "now"
    win = d[(d.date > anchor - timedelta(days=120)) & (d.date <= anchor)]
    win = win.dropna(subset=feats).drop_duplicates("berg_id", keep="last")
    if win.empty:
        raise RuntimeError("no iceberg rows with complete features")

    # Rank predicted-moving bergs first: a map of stationary markers shows
    # nothing about the drift model, which is the point of having one.
    pred_moving = bundle["clf"].predict(win[feats].values).astype(bool)
    win = win.assign(_moving=pred_moving).sort_values("_moving", ascending=False).head(max_bergs)

    X = win[feats].values
    moving = bundle["clf"].predict(X).astype(bool)
    u = np.zeros(len(X)); v = np.zeros(len(X))
    if moving.any():
        u[moving] = bundle["reg"]["u_berg"].predict(X[moving])
        v[moving] = bundle["reg"]["v_berg"].predict(X[moving])

    total_h = max((end_time - start_time).total_seconds() / 3600.0, step_hours)
    steps = int(total_h / step_hours) + 1

    out = []
    for i, (_, r) in enumerate(win.iterrows()):
        lat0, lon0 = float(r.lat), float(r.lon)
        spd_ms = float(np.hypot(u[i], v[i]))
        heading = float((np.degrees(np.arctan2(u[i], v[i])) + 360.0) % 360.0)

        traj = []
        for s in range(steps):
            secs = s * step_hours * 3600.0
            la = lat0 + v[i] * secs / 110570.0
            lo = lon0 + u[i] * secs / (111320.0 * max(np.cos(np.radians(lat0)), 1e-6))
            lo = ((lo + 180.0) % 360.0) - 180.0
            traj.append({
                "timestamp": (start_time + timedelta(hours=s * step_hours)).isoformat(),
                "lat": float(la), "lon": float(lo),
            })

        out.append({
            "id": str(r.berg_id).upper(),
            "name": f"Iceberg {str(r.berg_id).upper()}",
            "sizeCategory": _size_category(float(r.size_1)),
            "initialPosition": {"lat": lat0, "lon": lon0},
            "headingDeg": heading,
            "speedKnots": spd_ms * 1.94384,
            "currentSpeedKnots": float(np.hypot(r.u_curr, r.v_curr)) * 1.94384,
            "currentDirDeg": float((np.degrees(np.arctan2(r.u_curr, r.v_curr)) + 360.0) % 360.0),
            "windSpeedKnots": float(np.hypot(r.u_wind, r.v_wind)) * 1.94384,
            "windDirDeg": float((np.degrees(np.arctan2(r.u_wind, r.v_wind)) + 360.0) % 360.0),
            "trajectory": traj,
            "source": "two-stage GBM drift model (3.96 km RMS @24h)",
            "predictedMoving": bool(moving[i]),
        })
    return out


# ------------------------------------------------------- environment layers
# Currents and wind come from the SAME regridded daily archive the sea-ice
# model consumes, so every layer on the map is consistent with the forecast:
# channel order is [sic, wind_u, wind_v, air_temp, sst, current_u, current_v].

@lru_cache(maxsize=1)
def _sample_index(lat_step: float, lon_step: float):
    """Nearest PS-25 cell for a regular lat/lon sampling grid (Antarctic only)."""
    z = np.load(LONLAT_P)
    glon, glat = z["lons"], z["lats"]
    gl, gt = np.radians(glon.ravel()), np.radians(glat.ravel())
    G = np.stack([np.cos(gt) * np.cos(gl), np.cos(gt) * np.sin(gl), np.sin(gt)], 1)

    lats = np.arange(-80.0, 80.0, lat_step)
    lons = np.arange(-180.0, 180.0, lon_step)
    out = {}
    for la in lats:
        if la > -50.0:          # archive only covers the Southern Ocean
            continue
        t = np.radians(la)
        for lo in lons:
            p = np.radians(lo)
            v = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)])
            k = int(np.argmax(G @ v))
            out[(float(la), float(lo))] = (k // glon.shape[1], k % glon.shape[1])
    return lats, lons, out


@lru_cache(maxsize=4)
def _forcing_day(date_str: str) -> np.ndarray:
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    f = DAILY / str(d.year) / f"{d.strftime('%Y%m%d')}.npz"
    if not f.exists():
        raise FileNotFoundError(f"no regridded forcing for {d}")
    return np.load(f)["data"].astype(np.float32)


def _vector_layer(date_str, u_ch, v_ch, lat_step, lon_step,
                  min_speed_knots, strong, moderate, source) -> dict:
    ds = date_str or DEFAULT_FORECAST_DATE.strftime("%Y-%m-%d")
    arr = _forcing_day(ds)
    mask = np.load(MASK_P)
    lats, lons, idx = _sample_index(lat_step, lon_step)

    feats = []
    for (la, lo), (r, c) in idx.items():
        if mask[r, c] == 0:                       # land
            continue
        u_ms, v_ms = float(arr[u_ch, r, c]), float(arr[v_ch, r, c])
        if not (np.isfinite(u_ms) and np.isfinite(v_ms)):
            continue
        kn = round(float(np.hypot(u_ms, v_ms)) * 1.94384, 1)
        if kn <= min_speed_knots:
            continue
        feats.append({
            "type": "Feature",
            "properties": {
                "u": round(u_ms, 2), "v": round(v_ms, 2),
                "speedKnots": kn,
                "dirDeg": round(float(np.degrees(np.arctan2(u_ms, v_ms))) % 360.0, 1),
                "intensity": "strong" if kn > strong else "moderate" if kn > moderate else "light",
                "source": source, "valid_date": ds,
            },
            "geometry": {"type": "Point", "coordinates": [float(lo), float(la)]},
        })
    return {"type": "FeatureCollection", "features": feats}


def ocean_currents_geojson_real(date_str: str | None = None) -> dict:
    """Real GLORYS12 surface currents (channels 5, 6) as a GeoJSON vector field."""
    return _vector_layer(date_str, 5, 6, 5.0, 7.5, 0.2, 0.8, 0.4,
                         "GLORYS12 reanalysis (CMEMS)")


def wind_geojson_real(date_str: str | None = None) -> dict:
    """Real ERA5 10 m wind (channels 1, 2) as a GeoJSON vector field."""
    return _vector_layer(date_str, 1, 2, 5.5, 8.0, 1.0, 18.0, 10.0,
                         "ERA5 reanalysis (ECMWF)")


# ------------------------------------------------------------- ice risk grid
def ice_risk_geojson_real(date_str: str | None = None,
                          polar_class: str = "PC4") -> dict:
    """IMO POLARIS navigational risk derived from the SIC forecast."""
    ensure_seaice_importable()
    from seaice_forecast.risk.polaris import risk_ice_array

    ds = date_str or DEFAULT_FORECAST_DATE.strftime("%Y-%m-%d")
    sic = predict_sic(ds)
    risk = risk_ice_array(sic, polar_class=polar_class)
    lats, lons, idx = _grid_lookup()

    feats = []
    for la in lats:
        for lo in lons:
            r, c = idx[(float(la), float(lo))]
            rv = float(risk[r, c])
            if rv < 0.05:
                continue
            hlat, hlon = 0.75, 1.5
            feats.append({
                "type": "Feature",
                "properties": {
                    "risk": round(rv, 3),
                    "concentration": round(float(sic[r, c]) * 100.0, 1),
                    "severity": "severe" if rv >= 0.6 else "elevated" if rv >= 0.3 else "low",
                    "polarClass": polar_class,
                    "source": "IMO POLARIS on UNet+ConvLSTM forecast",
                    "valid_date": ds,
                },
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lo - hlon, la - hlat], [lo + hlon, la - hlat],
                    [lo + hlon, la + hlat], [lo - hlon, la + hlat],
                    [lo - hlon, la - hlat],
                ]]},
            })
    return {"type": "FeatureCollection", "features": feats}



# ------------------------------------------------- point lookup for routing
# The A* route engine samples sea-ice concentration per graph edge, hundreds of
# thousands of times per request. It needs a cheap scalar lookup, not GeoJSON,
# so we build a KD-tree over the PS-25 grid once and reuse it.

@lru_cache(maxsize=1)
def _kdtree():
    from scipy.spatial import cKDTree
    z = np.load(LONLAT_P)
    glon, glat = z["lons"], z["lats"]
    gl, gt = np.radians(glon.ravel()), np.radians(glat.ravel())
    pts = np.stack([np.cos(gt) * np.cos(gl), np.cos(gt) * np.sin(gl), np.sin(gt)], 1)
    return cKDTree(pts), glon.shape


@lru_cache(maxsize=2)
def _sic_flat(date_str: str):
    sic = predict_sic(date_str)
    return sic.ravel()


def sic_at_point(lat: float, lon: float, date_str: str | None = None) -> float:
    """Forecast sea-ice concentration (0-100 %) at a lat/lon.

    Point sea-ice concentration for the route coster, backed by
    the trained U-Net+ConvLSTM forecast. Returns 0.0 outside the model domain
    (north of ~50 S) so non-polar routing is unaffected.
    """
    if lat > -50.0:
        return 0.0
    ds = date_str or DEFAULT_FORECAST_DATE.strftime("%Y-%m-%d")
    tree, shape = _kdtree()
    t, p = np.radians(lat), np.radians(lon)
    v = np.array([[np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)]])
    _, k = tree.query(v, k=1)
    return float(_sic_flat(ds)[int(k[0])] * 100.0)


# ------------------------------------------------------------- model status
# Every endpoint returns an empty field on failure so the map never 500s.
# That is the right call for availability and the wrong one for trust: a working
# map does not prove the models loaded. This actually runs each model so the UI
# can say which of the two it is.

def model_status() -> dict[str, Any]:
    """Exercise each trained model once and report what really loaded."""
    artifacts = {
        "daily_archive": DAILY, "land_mask": MASK_P, "norm_stats": NORM_P,
        "grid_lonlat": LONLAT_P, "sic_checkpoint": SIC_CKPT,
        "drift_model": DRIFT_PKL, "drift_fixes": DRIFT_CSV,
    }
    missing = [n for n, p in artifacts.items() if not p.exists()]
    ds = DEFAULT_FORECAST_DATE.strftime("%Y-%m-%d")

    sea_ice: dict[str, Any] = {"live": False}
    try:
        sic = predict_sic(ds)
        sea_ice = {
            "live": True,
            "model": "U-Net + ConvLSTM (NSIDC CDR v6)",
            "validDate": ds,
            "gridShape": list(sic.shape),
            "meanConcentration": round(float(sic.mean()) * 100, 2),
            "iceCoveredPct": round(float((sic > 0.15).mean()) * 100, 2),
        }
    except Exception as exc:  # noqa: BLE001 - status must never raise
        sea_ice["error"] = f"{type(exc).__name__}: {exc}"

    drift: dict[str, Any] = {"live": False}
    try:
        bundle, frame = _drift_bundle()
        drift = {
            "live": True,
            "model": "two-stage GBM (3.96 km RMS @24h)",
            "bergFixes": int(len(frame)),
            "latestFix": frame["date"].max().strftime("%Y-%m-%d"),
        }
    except Exception as exc:  # noqa: BLE001
        drift["error"] = f"{type(exc).__name__}: {exc}"

    polaris: dict[str, Any] = {"live": False}
    try:
        ensure_seaice_importable()
        from seaice_forecast.risk.polaris import risk_ice
        polaris = {
            "live": True,
            "model": "IMO POLARIS risk index",
            "sample": {"sic": 0.7, "PC4": round(float(risk_ice(0.7, polar_class="PC4")), 3)},
        }
    except Exception as exc:  # noqa: BLE001
        polaris["error"] = f"{type(exc).__name__}: {exc}"

    components = {"seaIce": sea_ice, "icebergDrift": drift, "polarisRisk": polaris}
    live = [k for k, v in components.items() if v["live"]]
    return {
        "modelRoot": str(MODEL_ROOT),
        "allLive": len(live) == len(components),
        "liveComponents": live,
        "missingArtifacts": missing,
        "forecastDate": ds,
        "components": components,
    }
