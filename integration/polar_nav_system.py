#!/usr/bin/env python3
"""
Integrated Antarctic Navigation Decision Support System  (SIH PS 26059).

Chains all four components into one call:

    SIC forecast  ->  POLARIS ice risk  ->  fuel cost  ->  A* route
                  \
                   ->  iceberg drift forecast  ->  hazard positions

Each sub-model is loaded from its own trained artifact; nothing is retrained
here. The output is a single JSON-serialisable dict - that dict IS the
integration contract every consumer (UI, report, demo) should read.

Usage:
    venv/bin/python integration/polar_nav_system.py --date 2017-06-15 \
        --start 65 80 --goal 30 160 --polar-class PC4
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEAICE = ROOT / "seaice_forecast"
DRIFT = ROOT / "iceberg-drift"
sys.path.insert(0, str(SEAICE / "src"))

from seaice_forecast.risk.polaris import risk_ice_array            # noqa: E402
from seaice_forecast.fuel import fuel_per_cell_array               # noqa: E402
from seaice_forecast.fuel.consumption import speed_in_ice_array    # noqa: E402
from seaice_forecast.routing.pathfinding import (                  # noqa: E402
    build_cost_grid, astar_path, path_distance,
)

DAILY = SEAICE / "data/processed/regridded/daily"
MASK_P = SEAICE / "data/processed/land_ocean_mask_ps25.npy"
NORM_P = SEAICE / "data/processed/normalization_stats.json"
SIC_CKPT = SEAICE / "models/checkpoints/phase3/sic_unet_convlstm_v001_best.pt"
DRIFT_PKL = DRIFT / "models/drift_twostage.pkl"
DRIFT_CSV = DRIFT / "data/drift_trainable_rich.csv"

VARS = ["sic", "wind_u", "wind_v", "air_temp", "sst", "current_u", "current_v"]


class PolarNavigationSystem:
    """Loads every trained component once, then serves integrated forecasts."""

    def __init__(self, polar_class: str = "PC4", device: str | None = None):
        import torch
        self.polar_class = polar_class
        self.mask = np.load(MASK_P)
        self.norm = json.loads(NORM_P.read_text())
        self.torch = torch
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # --- sea-ice forecaster -------------------------------------------
        from seaice_forecast.models.unet_convlstm import UNetConvLSTM
        self.sic_model = UNetConvLSTM(
            in_channels=7, output_channels=1, seq_len=7,
            encoder_channels=[16, 32, 64, 128], convlstm_layers=1,
            dropout=0.0, output_activation="sigmoid",
        )
        ck = torch.load(SIC_CKPT, map_location="cpu")
        self.sic_model.load_state_dict(ck["model_state_dict"])
        self.sic_model.to(self.device).eval()

        # --- iceberg drift model ------------------------------------------
        import joblib
        self.drift = joblib.load(DRIFT_PKL)

    # ------------------------------------------------------------------ SIC
    def _load_day(self, d) -> np.ndarray:
        f = DAILY / f"{d.year}" / f"{d.strftime('%Y%m%d')}.npz"
        if not f.exists():
            raise FileNotFoundError(f"no regridded data for {d}")
        return np.load(f)["data"].astype(np.float32)

    def forecast_sic(self, date: datetime) -> dict:
        """7 days of history -> next-day SIC. Returns forecast and persistence."""
        hist = np.stack([self._load_day((date - timedelta(days=6 - i)).date())
                         for i in range(7)])          # [7,7,H,W]
        x = hist.copy()
        for c, v in enumerate(VARS):
            if v != "sic" and v in self.norm:
                s = self.norm[v]["std"]
                if s > 1e-6:
                    x[:, c] = (x[:, c] - self.norm[v]["mean"]) / s
        x[:, 1:, self.mask == 0] = 0.0                # land -> normalised mean

        with self.torch.no_grad():
            t = self.torch.from_numpy(x).unsqueeze(0).float().to(self.device)
            pred = self.sic_model(t).squeeze().cpu().numpy()
        pred = np.clip(pred, 0, 1)
        pred[self.mask == 0] = 0.0
        return {"sic_forecast": pred, "sic_persistence": hist[-1, 0]}

    # --------------------------------------------------------------- drift
    def forecast_drift(self, date: datetime, max_bergs: int = 25) -> list:
        """Predict 24 h displacement for bergs observed on/near `date`."""
        import pandas as pd
        d = pd.read_csv(DRIFT_CSV)
        d["date"] = pd.to_datetime(d["date"])
        mo = d["date"].dt.month
        d["month_s"] = np.sin(2 * np.pi * mo / 12)
        d["month_c"] = np.cos(2 * np.pi * mo / 12)
        d["wind_spd"] = np.hypot(d.u_wind, d.v_wind)
        d["curr_spd"] = np.hypot(d.u_curr, d.v_curr)

        feats = self.drift["features"]
        win = d[(d.date >= date - timedelta(days=10)) & (d.date <= date)]
        win = win.dropna(subset=feats).drop_duplicates("berg_id", keep="last")
        if win.empty:
            return []
        win = win.head(max_bergs)

        X = win[feats].values
        moving = self.drift["clf"].predict(X).astype(bool)
        u = np.zeros(len(X)); v = np.zeros(len(X))
        if moving.any():
            u[moving] = self.drift["reg"]["u_berg"].predict(X[moving])
            v[moving] = self.drift["reg"]["v_berg"].predict(X[moving])

        out = []
        for i, (_, r) in enumerate(win.iterrows()):
            dlat = v[i] * 86400 / 110570.0
            dlon = u[i] * 86400 / (111320.0 * np.cos(np.radians(r.lat)))
            out.append({
                "berg_id": str(r.berg_id),
                "lat": float(r.lat), "lon": float(r.lon),
                "pred_lat_24h": float(r.lat + dlat),
                "pred_lon_24h": float(r.lon + dlon),
                "u_ms": float(u[i]), "v_ms": float(v[i]),
                "drift_km_24h": float(np.hypot(u[i], v[i]) * 86400 / 1000),
                "moving": bool(moving[i]),
            })
        return out

    # -------------------------------------------------------------- routing
    def plan_route(self, sic, start, goal, distance_weight=1.0,
                   risk_weight=2.0, fuel_weight=0.0) -> dict:
        land = self.mask == 0
        cost = build_cost_grid(
            sic_grid=sic, polar_class=self.polar_class,
            distance_weight=distance_weight, risk_weight=risk_weight,
            fuel_weight=fuel_weight, land_mask=land, pixel_size_km=25.0,
        )

        path = astar_path(cost, tuple(start), tuple(goal))
        if not path:
            return {"found": False}

        rr = np.array([p[0] for p in path]); cc = np.array([p[1] for p in path])
        s = sic[rr, cc]
        fuel_cells = fuel_per_cell_array(25.0, s, polar_class=self.polar_class)
        spd = speed_in_ice_array(s, polar_class=self.polar_class)
        return {
            "found": True,
            "waypoints": [[int(a), int(b)] for a, b in path],
            "n_waypoints": len(path),
            "distance_km": float(path_distance(path, 25.0)),
            "fuel_tonnes": float(fuel_cells.sum()),
            "transit_hours": float((25.0 / (spd * 1.852)).sum()),
            "mean_sic_on_route": float(s.mean()),
            "max_sic_on_route": float(s.max()),
            "mean_risk_on_route": float(
                risk_ice_array(s, polar_class=self.polar_class).mean()),
        }

    # ------------------------------------------------------------------ run
    def run(self, date: datetime, start, goal) -> dict:
        f = self.forecast_sic(date)
        sic = f["sic_forecast"]
        ocean = self.mask == 1
        risk = risk_ice_array(sic, polar_class=self.polar_class)
        return {
            "schema_version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "valid_date": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
            "polar_class": self.polar_class,
            "grid": {"shape": list(sic.shape), "projection": "EPSG:3412",
                     "pixel_size_km": 25.0},
            "sea_ice": {
                "mean_sic_ocean": float(sic[ocean].mean()),
                "ice_covered_pct": float(100 * (sic[ocean] > 0.15).mean()),
                "max_sic": float(sic.max()),
            },
            "risk": {
                "mean_risk_ocean": float(risk[ocean].mean()),
                "high_risk_pct": float(100 * (risk[ocean] > 0.5).mean()),
            },
            "icebergs": self.forecast_drift(date),
            "routes": {
                "shortest": self.plan_route(sic, start, goal, 1.0, 0.0, 0.0),
                "safest": self.plan_route(sic, start, goal, 1.0, 2.0, 0.0),
                "fuel_optimal": self.plan_route(sic, start, goal, 1.0, 0.0, 2.0),
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2017-06-15")
    ap.add_argument("--start", type=int, nargs=2, default=[65, 80])
    ap.add_argument("--goal", type=int, nargs=2, default=[30, 160])
    ap.add_argument("--polar-class", default="PC4")
    ap.add_argument("--out", default=str(ROOT / "integration/output/nav_forecast.json"))
    a = ap.parse_args()

    sysm = PolarNavigationSystem(polar_class=a.polar_class)
    res = sysm.run(datetime.strptime(a.date, "%Y-%m-%d"), a.start, a.goal)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    print("=" * 66)
    print(f"INTEGRATED FORECAST  valid {res['valid_date']}  class {res['polar_class']}")
    print("=" * 66)
    si = res["sea_ice"]
    print(f"sea ice   mean SIC {si['mean_sic_ocean']:.3f} | ice-covered {si['ice_covered_pct']:.1f}%")
    print(f"risk      mean {res['risk']['mean_risk_ocean']:.3f} | high-risk area {res['risk']['high_risk_pct']:.1f}%")
    print(f"icebergs  {len(res['icebergs'])} tracked")
    for b in res["icebergs"][:3]:
        print(f"            {b['berg_id']:<8} drift {b['drift_km_24h']:5.1f} km/24h")
    print(f"\n{'route':<14}{'dist km':>10}{'fuel t':>10}{'hours':>9}{'mean SIC':>10}")
    for name, r in res["routes"].items():
        if r.get("found"):
            print(f"{name:<14}{r['distance_km']:>10.0f}{r['fuel_tonnes']:>10.1f}"
                  f"{r['transit_hours']:>9.1f}{r['mean_sic_on_route']:>10.3f}")
        else:
            print(f"{name:<14}{'NO ROUTE FOUND':>39}")
    print("=" * 66)
    print(f"written -> {a.out}")


if __name__ == "__main__":
    main()
