#!/usr/bin/env python3
"""Build one loadable .pkl per navigation component.

    python scripts/export_model_pickles.py

Writes to <model root>/model_exports/. Each pickle contains a predictor object
with a `predict()` method plus metadata saying whether it holds fitted weights
or the constants of a published formula, and what it needs to load.

Run from the backend directory with the model environment's interpreter:
    ../../seaice_forecast/venv/bin/python scripts/export_model_pickles.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from app.model_artifacts import (  # noqa: E402
    FuelConsumptionPredictor,
    IcebergDriftPredictor,
    PolarisRiskPredictor,
    SeaIceForecastPredictor,
)
from app.model_bridge import (  # noqa: E402
    DRIFT_PKL, MASK_P, MODEL_ROOT, SEAICE, SIC_CKPT,
    _load_norm_stats, ensure_seaice_importable,
)

OUT_DIR = MODEL_ROOT / "model_exports"
EVAL_DRIFT = MODEL_ROOT / "iceberg-drift/output/drift_twostage_metrics.json"
EVAL_SEAICE = SEAICE / "output/evaluation/baseline_comparison.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def build_polaris() -> PolarisRiskPredictor:
    ensure_seaice_importable()
    from seaice_forecast.risk.polaris import DEFAULT_POLARIS_CONFIG as cfg
    return PolarisRiskPredictor(
        class_exponents=cfg.class_exponents,
        open_water_threshold=cfg.open_water_threshold,
        open_water_risk=cfg.open_water_risk,
        max_risk=cfg.max_risk,
    )


def build_fuel() -> FuelConsumptionPredictor:
    ensure_seaice_importable()
    from seaice_forecast.fuel.consumption import DEFAULT_FUEL_CONFIG as cfg
    from seaice_forecast.fuel.consumption import HEAVY_ICE_SPEED_RETENTION
    return FuelConsumptionPredictor(
        retention=HEAVY_ICE_SPEED_RETENTION,
        base_rate_t_per_km=cfg.base_rate_t_per_km,
        v_ref_kn=cfg.v_ref_kn,
        max_power_ratio=cfg.max_power_ratio,
        power_ramp_k=cfg.power_ramp_k,
        sic_exponent=cfg.sic_exponent,
        heavy_ice_sic=cfg.heavy_ice_sic,
    )


def build_drift() -> IcebergDriftPredictor:
    bundle = joblib.load(DRIFT_PKL)
    metrics = _load_json(EVAL_DRIFT).get("two_stage", {})
    return IcebergDriftPredictor(
        clf=bundle["clf"], reg=bundle["reg"], features=bundle["features"], metrics=metrics,
    )


def build_seaice() -> SeaIceForecastPredictor:
    import torch
    ensure_seaice_importable()
    ck = torch.load(SIC_CKPT, map_location="cpu")
    arch = dict(
        in_channels=7, output_channels=1, seq_len=7,
        encoder_channels=[16, 32, 64, 128], convlstm_layers=1,
        dropout=0.0, output_activation="sigmoid",
    )
    return SeaIceForecastPredictor(
        state_dict=ck["model_state_dict"],
        arch=arch,
        norm=_load_norm_stats(),
        mask=np.load(MASK_P),
        metrics=_load_json(EVAL_SEAICE),
    )


BUILDERS = {
    "polaris_risk": build_polaris,
    "fuel_consumption": build_fuel,
    "iceberg_drift": build_drift,
    "seaice_forecast": build_seaice,
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for key, build in BUILDERS.items():
        path = OUT_DIR / f"{key}.pkl"
        print(f"[export] {key} ...", flush=True)
        try:
            obj = build()
            joblib.dump(obj, path, compress=3)
        except Exception as exc:  # noqa: BLE001
            print(f"[export] FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
            continue

        # Load it back and use it. An artifact that cannot round-trip is not an
        # artifact, and that is only visible if you actually call predict().
        reloaded = joblib.load(path)
        if key == "polaris_risk":
            sample = {"input": "SIC 0.70, PC4", "output": round(float(reloaded.predict(0.70, "PC4")), 4)}
        elif key == "fuel_consumption":
            r = reloaded.predict(0.70, "PC4", 25.0)
            sample = {"input": "SIC 0.70, PC4, 25 km cell",
                      "output": {k: round(float(v), 3) for k, v in r.items()}}
        elif key == "iceberg_drift":
            import pandas as pd
            d = pd.read_csv(MODEL_ROOT / "iceberg-drift/data/drift_trainable_rich.csv")
            mo = pd.to_datetime(d["date"]).dt.month
            d["month_s"] = np.sin(2 * np.pi * mo / 12); d["month_c"] = np.cos(2 * np.pi * mo / 12)
            d["wind_spd"] = np.hypot(d.u_wind, d.v_wind); d["curr_spd"] = np.hypot(d.u_curr, d.v_curr)
            usable = d.dropna(subset=reloaded.features).head(500)
            # Row 0 is a grounded berg, so the sample used to read "0.0 km" -
            # a correct answer that makes a working model look broken. Show one
            # the classifier actually gates as moving.
            gate = reloaded.predict(usable)["moving"]
            hits = [i for i, m in enumerate(gate) if m]
            row = usable.iloc[[hits[0] if hits else 0]]
            r = reloaded.predict(row)
            sample = {"input": f"NIC berg {row['berg_id'].iloc[0]} on {row['date'].iloc[0]}",
                      "output": {"moving": bool(r["moving"][0]),
                                 "drift_km_24h": round(float(r["drift_km_24h"][0]), 3)}}
        else:
            from app.model_bridge import DAILY
            from datetime import timedelta
            date = datetime(2018, 12, 20)
            hist = np.stack([
                np.load(DAILY / str((date - timedelta(days=6 - i)).year) /
                        f"{(date - timedelta(days=6 - i)).strftime('%Y%m%d')}.npz")["data"]
                for i in range(7)
            ])
            out = reloaded.predict(hist)
            sample = {"input": "7 days ending 2018-12-20",
                      "output": {"shape": list(out.shape), "mean": round(float(out.mean()), 4),
                                 "max": round(float(out.max()), 4)}}

        entry = {
            "key": key,
            "file": path.name,
            "name": reloaded.name,
            "kind": reloaded.kind,
            "source": reloaded.source,
            "requires": reloaded.requires,
            "output": reloaded.output,
            "sizeMB": round(path.stat().st_size / 1e6, 2),
            "sha256": _sha256(path),
            "roundTrip": sample,
        }
        manifest.append(entry)
        print(f"          {entry['sizeMB']} MB  sha256={entry['sha256'][:16]}…  predict() -> {sample['output']}")

    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "generatedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "loadWith": "joblib.load(path)  # then .predict(...)",
        "note": (
            "Two artifacts hold weights fitted from data (seaice_forecast, "
            "iceberg_drift); two hold the constants of a published formula "
            "(polaris_risk, fuel_consumption) and have no learned parameters."
        ),
        "artifacts": manifest,
    }, indent=2))

    print(f"\n[export] {len(manifest)}/{len(BUILDERS)} written to {OUT_DIR}")
    return 0 if len(manifest) == len(BUILDERS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
